import logging

import mxnet as mx


def get_create_checkpoint_callback(iteration, model_prefix):

    def create_checkpoint(execution_params):
        if execution_params.nbatch % iteration == 0:
            original_executor = execution_params.locals['executor_manager']
            save_dict = {('arg:%s' % k): v[0].as_in_context(mx.cpu()) for k, v in zip(original_executor.param_names, original_executor.param_arrays)}
            save_dict.update({('aux:%s' % k): v[0].as_in_context(mx.cpu()) for k, v in zip(original_executor.aux_names, original_executor.aux_arrays)})

            symbol = execution_params.locals['symbol']
            symbol.save('{}-symbol.json'.format(model_prefix))

            model_name = "{}-{:0>4}-{:0>5}".format(model_prefix, execution_params.epoch, execution_params.nbatch)
            mx.nd.save(
                model_name,
                save_dict,
            )
            logging.info('Saved checkpoint to \"{}\"'.format(model_name))

    return create_checkpoint
