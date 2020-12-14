def i_scalar_pretty_print(message, scalar_name, scalar):

    message += scalar_name + " = %i"
    print(message % scalar)


def f_scalar_pretty_print(message, scalar_name, scalar):

    message += scalar_name + " = %.4E"
    print(message % scalar)


def i_array_pretty_print(message, array_name, array):

    message += array_name + " = [{:s}]"
    print(message.format(", ".join([str(a) for a in array])))


def f_array_pretty_print(message, array_name, array):

    message += array_name + " = [{:s}]"
    print(message.format(", ".join(["{:.16e}".format(a) for a in array])))


def sym_array_pretty_print(message, array_name, array):

    message += array_name + " = [{:s}]"
    print(message.format(", ".join([str(a) for a in array])))
