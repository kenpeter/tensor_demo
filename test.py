import tensorflow as tf

def main():
    # explode
    # y = w * x + b
    # y_pred = 

    tf.compat.v1.disable_eager_execution()

    # model param
    w = tf.Variable(tf.random.normal([1]))
    b = tf.Variable(tf.random.normal([1]))

    # define train input and output
    x = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
    y = tf.Variable([7, 12, 15, 14, 18], dtype=tf.float32)

    # this is model
    y_pred = tf.multiply(w, x) + b

    # set up expectation
    loss = tf.reduce_mean(tf.square(y_pred - y))

    # init tensor flow
    init = tf.compat.v1.global_variables_initializer()

    # training with optimizer + loss
    opt = tf.compat.v1.train.GradientDescentOptimizer(0.01);
    train_op = opt.minimize(loss)

    # tensor section
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        # loop times
        for _ in range(1000):
            _, loss_value = sess.run([train_op, loss])
    
        print("w: ", sess.run(w))
        print("b: ", sess.run(b))

if __name__ == "__main__": 
    print('hi')
    main()