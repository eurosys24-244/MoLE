import mole.config as mole_config

class MoEMoniter():
    def __init__(self):
        self.expert_pool = {}
        self.done = False
        self.overflow = False

    def append_expert(self, layer_id, expert_id, expert):
        assert self.done == False, "not appending experts after building done"
        self.expert_pool[(layer_id, expert_id)] = expert

    def build_done(self):
        mole_config.MOLE_ACTIVATED = True
        self.done = True

    @property
    def sync_this_step(self):
        return self._sync_this_step

    @sync_this_step.setter
    def sync_this_step(self, sync_this_step):
        self._sync_this_step = sync_this_step
    
    @property
    def loss_scale(self):
        return self._loss_scale
    
    @loss_scale.setter
    def loss_scale(self, loss_scale):
        self._loss_scale = loss_scale

    def set_overflow(self, overflow):
        self.overflow = overflow

Moniter = MoEMoniter()
