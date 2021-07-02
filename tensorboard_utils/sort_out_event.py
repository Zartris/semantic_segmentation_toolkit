from pathlib import Path

import tensorflow as tf
from tensorflow.core.util import event_pb2
from torch.utils.tensorboard import SummaryWriter
from tensorflow.python.summary.summary_iterator import summary_iterator
if __name__ == '__main__':
    event_file_path = Path(
        "G:\\code\\python\\Hardnet\\BiSeNet\\runs\\BiSeNetv2\\cur\\events.out.tfevents.1610891663.DESKTOP-84E29NP")
    event_file_new_path = Path(
        "G:\\code\\python\\Hardnet\\BiSeNet\\runs\\BiSeNetv2\\cur\\events.out.tfevents.1610891663_fixed.DESKTOP-84E29NP")


    def get_file_name(counter):
        return Path(
            "G:\\code\\python\\Hardnet\\BiSeNet\\runs\\BiSeNetv2\\cur\\events.out.tfevents.1610891663_{counter}.DESKTOP-84E29NP")


    serialized_examples = tf.data.TFRecordDataset(str(event_file_path))
    # shards = 10
    #
    # for i in range(shards):
    #     writer = tf.data.experimental.TFRecordWriter(
    #         f"G:\\code\\python\\Hardnet\\BiSeNet\\runs\\BiSeNetv2\\cur\\events.out.tfevents_{i}.1610891663.DESKTOP-84E29NP")
    #     shard = serialized_examples.shard(shards, i)
    #     writer.write(serialized_examples.shard(shards, i))
    #
    log = 1000
    counter = -1
    for index, serialized_example in enumerate(serialized_examples):
        if index % log == 0:
            counter += 1
            writer = SummaryWriter(str(get_file_name(counter)))
        event = event_pb2.Event.FromString(serialized_example.numpy())
        event_type = event.WhichOneof('what')
        if event_type != 'summary':
            debug = 0
            # writer.(event)
            # writer.flush()
        else:
            wall_time = event.wall_time
            step = event.step

            # possible types: simple_value, image, histo, audio
            filtered_values = [value for value in event.summary.value if value.HasField('simple_value')]
            for v in filtered_values:
                print(wall_time, step, v.tag)
            debug = 0
            # summary = tf.Summary(value=filtered_values)
            #
            # filtered_event = tf.summary.Event(summary=summary,
            #                                   wall_time=wall_time,
            #                                   step=step)
            # writer.add_event(filtered_event)
        # for value in event.summary.value:
        #     t = tf.make_ndarray(value.tensor)
        #     print(value.tag, event.step, t, type(t))
    # for event in tf.python.summary.su(event_file_path):
    #     event_type = event.WhichOneof('what')
    # if event_type != 'summary':
    #     writer.add_event(event)
    # else:
    #     wall_time = event.wall_time
    #     step = event.step
    #
    #     # possible types: simple_value, image, histo, audio
    #     filtered_values = [value for value in event.summary.value if value.HasField('simple_value')]
    #     summary = tf.Summary(value=filtered_values)
    #
    #     filtered_event = tf.summary.Event(summary=summary,
    #                                       wall_time=wall_time,
    #                                       step=step)
    #     writer.add_event(filtered_event)
