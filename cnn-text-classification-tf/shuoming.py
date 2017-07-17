第一步：加载数据
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
返回的数据格式为：[(每个句子对应的单词),()，....]  和 [[1,0],[1,0],......,[0,1]]

第二步：产生批数据：
batches = data_helpers.batch_iter(
    list(zip(x_train, y_train)), FLAGS.batch_size(64), FLAGS.num_epochs(200))





第三步：运行图
1、def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)

2、 def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)

cnn = TextCNN(
    sequence_length=x_train.shape[1],
    num_classes=y_train.shape[1],
    vocab_size=len(vocab_processor.vocabulary_),
    embedding_size=FLAGS.embedding_dim,
    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
    num_filters=FLAGS.num_filters,
    l2_reg_lambda=FLAGS.l2_reg_lambda)