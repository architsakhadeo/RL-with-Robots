# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import builtins
import tempfile, zipfile
import numpy as np


def create_callback(shared_returns, save_model_basepath, load_model_path=None):
    builtins.shared_returns = shared_returns
    builtins.save_model_basepath = save_model_basepath
    builtins.load_model_path = load_model_path

    def kindred_callback(locals, globals):
        import tensorflow as tf
        saver = tf.train.Saver()

        shared_returns = globals['__builtins__']['shared_returns']
        savebasepath = globals['__builtins__']['save_model_basepath']
        if locals['iters_so_far'] == 0:
            loadpath = globals['__builtins__']['load_model_path']
            if loadpath is not None:
                saver.restore(tf.get_default_session(), loadpath)
        else:
            ep_rets = locals['seg']['ep_rets']
            ep_lens = locals['seg']['ep_lens']
            ep_ss = locals['seg']['ep_ss']
            if len(ep_rets):
                if not shared_returns is None:
                    shared_returns['write_lock'] = True
                    shared_returns['episodic_returns'] += ep_rets
                    shared_returns['episodic_lengths'] += ep_lens
                    shared_returns['episodic_ss'] += ep_ss
                    shared_returns['write_lock'] = False
                    np.save(savebasepath+'data/ep_lens',
                            np.array(shared_returns['episodic_lengths']))
                    np.save(savebasepath+'data/ep_rets',
                            np.array(shared_returns['episodic_returns']))
                    np.save(savebasepath+'data/ep_ss',
                            np.array(shared_returns['episodic_ss']))
        fname = savebasepath+'/models/' + str(locals['iters_so_far']) + '.ckpt'
        saver.save(tf.get_default_session(), fname)
    return kindred_callback
