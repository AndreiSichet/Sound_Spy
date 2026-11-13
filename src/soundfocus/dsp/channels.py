"""``ChannelReducer``: the only place where channels are reduced.

Turns a multichannel ``AudioBuffer`` into the channel layout a consumer needs (modes:
mean downmix, select channel; beamforming toward a direction in module 3). No other code
may downmix or drop channels.

TODO(step-04): implement.
TODO(module-3): implement.
"""
