import tensorflow as tf

def main():
    # explode

    # turn off eager
    tf.compat.v1.disable_eager_execution()

    # input and output
    x = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
    y = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)

    # model param with init
    rad = tf.Variable(tf.random.normal([1]))
    balance = tf.Variable(tf.random.normal([1]))
    

    # input x, y to train better model param 
    # model === prediction
    y_pred = tf.multiply(rad, x) + balance

    # reduce loss, so better model param
    loss = tf.reduce_mean(tf.square(y_pred - y))

    # train op
    # init op
    init_op = tf.compat.v1.global_variables_initializer()

    # train with optimizer
    # then with min loss
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    # train op not a func, but param
    train_op = optimizer.minimize(loss)

    # all tf session()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        for _ in range(1000):
            _, loss_val = sess.run([train_op, loss])

        # rad ts obj; sess run ts obj will output
        print("rad: ", sess.run(rad))
        print("balance: ", sess.run(balance))



# need to sit on every left, then it will run
if __name__ == "__main__":
    print('hi')
    main()