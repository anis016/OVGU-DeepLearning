import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * x + y + 2 # f(x) = x^3 + y + 2

# prepare an init node to initialize all the variables
init = tf.global_variables_initializer()
with tf.Session() as session: # initialize session with ContextManager
    session.graph.finalize() # finalize the graph
    init.run() # initialize all the variables
    result = f.eval()

    print(result)