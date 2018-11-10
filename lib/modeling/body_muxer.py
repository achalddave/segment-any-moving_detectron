import abc

import torch
from torch import nn


def assert_all_equal(lst, error_prefix=''):
    if any(x != lst[0] for x in lst[1:]):
        if error_prefix and error_prefix[:-1] != ' ':
            error_prefix = error_prefix + ' '
        elif not error_prefix:
            error_prefix = 'All values not equal: '
        raise ValueError('%s%s' % (error_prefix, lst))
    return True


class BodyMuxer(nn.Module, abc.ABC):
    # BodyMuxer should not be directly constructed. Instead, use
    # BodyMuxer_<merger>, e.g., BodyMuxer_Average.
    @abc.abstractmethod
    def __init__(self, conv_bodies, conv_body_inputs):
        """
        Args:
            conv_bodies (list): List of functions, each of which creates a conv
                body.
            conv_body_inputs (list): List of length len(conv_bodies). Each
                element is a list of input indices indicating the inputs to
                pass to the corresponding conv_body.
        """
        super().__init__()
        assert len(conv_body_inputs) == len(conv_bodies)
        self.bodies = nn.ModuleList(
            [body_fn() for i, body_fn in enumerate(conv_bodies)])

        assert_all_equal([body.spatial_scale for body in self.bodies],
                         'Spatial scales of bodies do not match:')
        self.spatial_scale = self.bodies[0].spatial_scale

        # List of length len(conv_bodies). Each element is a list of channels
        # indicating the inputs to pass to the corresponding conv_body.
        self.body_channels = self._inputs_to_channels(conv_body_inputs)
        self.mapping_to_detectron = None
        self.merged_dim_out = None
        self.bodies_dim_out = [b.dim_out for b in self.bodies]

    @property
    def dim_out(self):
        assert self.merged_dim_out is not None
        return self.bodies_dim_out + [self.merged_dim_out]

    @staticmethod
    def _inputs_to_channels(inputs):
        body_channels = []
        for input_indices in inputs:
            selected_channels = []
            for i in input_indices:
                selected_channels += (3 * i, 3 * i + 1, 3 * i + 2)
            body_channels.append(selected_channels)
        return body_channels

    def _body_key(self, i):
        return 'body_%s' % i

    @abc.abstractmethod
    def _merge(self, outputs):
        pass

    def detectron_weight_mapping(self):
        # Copied from Generalized_RCNN.detectron_weight_mapping.
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.bodies.named_children():
                new_name = 'bodies.' + name
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = (
                        m_child.detectron_weight_mapping())
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = new_name + '.' + key
                        d_wmap[new_key] = value

            for name, m_child in self.named_children():
                if name == 'bodies':
                    continue
                # We need to have each module in the d_wmap for loading a
                # checkpoint using utils.net.load_ckpt, even if there is no
                # detectron equivalent key.
                for key in m_child.state_dict():
                    d_wmap[name + '.' + key] = '# NO_DETECTRON_EQUIVALENT'
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def forward(self, inputs):
        """
        Args:
            inputs (np.ndarray): Shape
                (batch_size, 3 * DATA_LOADER.NUM_INPUTS, w, h).
        """
        outputs = [
            body(inputs[:, selected_channels, :, :]) for body,
            selected_channels in zip(self.bodies, self.body_channels)
        ]

        if isinstance(outputs[0], list):
            # FPN, concatenate every corresponding level in outputs.
            num_levels = len(outputs[0])
            for i, output in enumerate(outputs[1:]):
                assert len(output) == num_levels, (
                    'Different number of FPN outputs in body %i and body 0 '
                    '(%s vs %s)' % (i+1, len(output), num_levels))

            concatenated_outputs = [
                self._merge([output[level] for output in outputs])
                for level in range(num_levels)
            ]
        else:
            concatenated_outputs = self._merge(outputs)
        return outputs + [concatenated_outputs]


class BodyMuxer_Average(BodyMuxer):
    def __init__(self, conv_bodies, conv_body_inputs):
        super().__init__(conv_bodies, conv_body_inputs)
        self.merged_dim_out = self.bodies[0].dim_out

    def _merge(self, outputs):
        """
        Args:
            outputs (list): Each element is an array of shape
                (num_images, num_channels, width, height)
        """
        assert_all_equal([x.shape for x in outputs],
                         'Shapes of outputs do not match: ')
        return torch.stack(outputs).mean(dim=0)


class BodyMuxer_Concatenate(BodyMuxer):
    def __init__(self, conv_bodies, conv_body_inputs):
        super().__init__(conv_bodies, conv_body_inputs)
        self.merged_dim_out = sum(body.dim_out for body in self.bodies)

    def _merge(self, outputs):
        """
        Args:
            outputs (list): Each element is an array of shape
                (num_images, num_channels, width, height)
        """
        # Only check for shapes of axes 0, 2, 3
        assert_all_equal([x[:, 0].shape for x in outputs],
                         'Shape mismatch in outputs: ')
        return torch.cat(outputs, dim=1)


class BodyMuxer_ConcatenateConv(BodyMuxer_Concatenate):
    def __init__(self, conv_bodies, conv_body_inputs, output_channels=None):
        super().__init__(conv_bodies, conv_body_inputs)
        if output_channels is None:
            output_channels = self.bodies[0].dim_out
        input_channels = sum(x.dim_out for x in self.bodies)
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1)
        self.merged_dim_out = output_channels

    def _merge(self, outputs):
        concatenated = super()._merge(outputs)
        return self.conv(concatenated)

    def init_conv_select_index_(self, body_index):
        """Initialize conv to select the outputs of a specific body."""
        body = self.bodies[body_index]
        assert self.merged_dim_out == body.dim_out

        # Input channel start_index corresponding to outputs of the specified
        # body's outputs.
        start_index = sum(x.dim_out for x in self.bodies[:body_index])
        kw, kh = self.conv.kernel_size

        assert kw % 2 == 1
        assert kh % 2 == 1
        mid_w = (kw - 1) / 2
        mid_h = (kh - 1) / 2
        assert mid_w.is_integer()
        assert mid_h.is_integer()
        mid_w = int(mid_w)
        mid_h = int(mid_h)

        # Shape (output_channels, input_channels, w, h)
        weight = self.conv.weight.data
        weight.zero_()
        for o in range(body.dim_out):
            weight[o, start_index+o, mid_w, mid_h] = 1

        self.conv.bias.data.zero_()


class BodyMuxer_ConcatenateAdapt(BodyMuxer_Concatenate):
    def __init__(self,
                 conv_bodies,
                 conv_body_inputs,
                 adaptor_name,
                 output_channels=None):
        super().__init__(conv_bodies, conv_body_inputs)
        self.adaptor_name = adaptor_name
        try:
            self.adaptor_fn = getattr(BodyMuxer_ConcatenateAdapt, adaptor_name)
            if not callable(self.adaptor_fn):
                raise AttributeError
        except AttributeError:
            # Caught AttributeError either because of getattr or because not
            # callable
            raise ValueError('Unknown adaptation function: %s' % adaptor_name)

        input_channels = sum(x.dim_out for x in self.bodies)
        if output_channels is None:
            output_channels = self.bodies[0].dim_out
        self.adaptor = self.adaptor_fn(input_channels, output_channels)
        self.merged_dim_out = output_channels

    def _merge(self, outputs):
        concatenated = super()._merge(outputs)
        return self.adaptor(concatenated)

    @staticmethod
    def adapt_3convs3x3(dim_in, dim_out):
        return BodyMuxer_ConcatenateAdapt.get_Xconvs3x3(dim_in, dim_out, 3)

    @staticmethod
    def get_Xconvs3x3(dim_in, dim_out, num_convs):
        module_list = []
        for i in range(num_convs):
            module_list.extend((
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)))
            dim_in = dim_out
        return nn.Sequential(*module_list)


class BodyMuxer_Difference(BodyMuxer):
    """Use a single backbone on all inputs and output consecutive differences.

    This module maintains a single conv_body. During the forward pass, the
    conv_body is applied to each of the conv_body_inputs. The output is a
    concatenation of the differences between the outputs corresponding to
    consecutive conv_body_inputs.
    """
    def __init__(self, conv_bodies, conv_body_inputs):
        assert_all_equal(
            conv_bodies,
            '{name} relies on a single, shared conv_body, but conv_bodies do '
            'not match:'.format(name=type(self).__name__)
        )
        nn.Module.__init__(self)
        self.body = conv_bodies[0]()
        self.merged_dim_out = self.body.dim_out * (len(conv_body_inputs) - 1)
        self.spatial_scale = self.body.spatial_scale
        self.body_channels = self._inputs_to_channels(conv_body_inputs)

        self.mapping_to_detectron = None

    def detectron_weight_mapping(self):
        # Copied from Generalized_RCNN.detectron_weight_mapping.
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            if list(self.body.parameters()):  # if module has any parameter
                child_map, child_orphan = (
                    self.body.detectron_weight_mapping())
                d_orphan.extend(child_orphan)
                for key, value in child_map.items():
                    new_key = 'body.' + key
                    d_wmap[new_key] = value

            for name, m_child in self.named_children():
                if name == 'body':
                    continue
                # We need to have each module in the d_wmap for loading a
                # checkpoint using utils.net.load_ckpt, even if there is no
                # detectron equivalent key.
                for key in m_child.state_dict():
                    d_wmap[name + '.' + key] = '# NO_DETECTRON_EQUIVALENT'
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _merge(self, outputs):
        """
        Args:
            outputs (list): Each element is an array of shape
                (num_images, num_channels, width, height), where num_channels
                is self.conv_bodies[0].dim_out.

        Returns:
            outputs (np.ndarray): Shape (num_images, concat_channels, width,
                height). concat_channels is
                (len(self.conv_bodies) - 1) * num_channels.
        """
        assert_all_equal([x.shape for x in outputs],
                         '[Likely an implementation bug] Shapes of outputs do '
                         'not match: ')
        output = torch.cat(outputs, dim=1)
        return (
            output[:, self.merged_dim_out:] - output[:, :-self.merged_dim_out])

    def forward(self, inputs):
        """
        Args:
            inputs (np.ndarray): Shape
                (batch_size, 3 * DATA_LOADER.NUM_INPUTS, w, h).
        """
        # This code is heavily duplicated from BodyMuxer to allow for a single
        # body to be run on multiple inputs.
        outputs = [
            self.body(inputs[:, selected_channels, :, :])
            for selected_channels in self.body_channels
        ]

        if isinstance(outputs[0], list):
            # FPN, concatenate every corresponding level in outputs.
            num_levels = len(outputs[0])
            for i, output in enumerate(outputs[1:]):
                assert len(output) == num_levels, (
                    'Different number of FPN outputs in body %i and body 0 '
                    '(%s vs %s)' % (i+1, len(output), num_levels))

            concatenated_outputs = [
                self._merge([output[level] for output in outputs])
                for level in range(num_levels)
            ]
        else:
            concatenated_outputs = self._merge(outputs)
        return outputs + [concatenated_outputs]


class BodyMuxer_DifferenceConcatenateConv(BodyMuxer_Difference):
    """Concatenate first output with successive differences, then apply a conv.

    This is like BodyMuxer_Difference, except instead of outputting the
    concatenation of
        outputs[1] - outputs[0], ...
    we first concatenate
        outputs[0], outputs[1] - outputs[0], ...
    and then apply a conv to reduce the dimensionality to self.body.dim_out.
    """
    def __init__(self, conv_bodies, conv_body_inputs):
        super().__init__(conv_bodies, conv_body_inputs)
        self.merged_dim_out = self.body.dim_out
        input_channels = self.body.dim_out * len(conv_body_inputs)
        self.conv = nn.Conv2d(
            input_channels, self.merged_dim_out, kernel_size=3, padding=1)

    def _merge(self, outputs):
        """
        Args:
            outputs (list): Each element is an array of shape
                (num_images, num_channels, width, height), where num_channels
                is self.conv_bodies[0].dim_out.

        Returns:
            outputs (np.ndarray): Shape (num_images, concat_channels, width,
                height). concat_channels is
                (len(self.conv_bodies)) * num_channels.
        """
        assert_all_equal([x.shape for x in outputs],
                         '[Likely an implementation bug] Shapes of outputs do '
                         'not match: ')
        output = torch.cat(outputs, dim=1)
        first_output = output[:, :self.merged_dim_out]
        differences = (
            output[:, self.merged_dim_out:] - output[:, :-self.merged_dim_out])
        output = torch.cat([first_output, differences], dim=1)
        return self.conv(output)
