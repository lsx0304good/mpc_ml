import random
import numpy as np

BASE = 10

# PRECISION_INTEGRAL = 1
# PRECISION_FRACTIONAL = 6
# Q = 10000019

PRECISION_a = 8
PRECISION_b = 8
Q = 293973345475167247070445277780365744413

PRECISION = PRECISION_a + PRECISION_b

assert (Q > BASE ** PRECISION)


def encode(r):
    tmp_coding = int(r * BASE ** PRECISION_b)
    field_element = tmp_coding % Q
    return field_element


def decode(field_element):
    tmp_coding = field_element if field_element <= Q / 2 else field_element - Q
    r = tmp_coding / BASE ** PRECISION_b
    return r

# secret sharing
def share(secret):
    first = random.randrange(Q)
    middle = random.randrange(Q)
    last = (secret - first - middle) % Q
    return [first, middle, last]


def reconstruct(sharing):
    return sum(sharing) % Q


def re_share(x):
    Y = [share(x[0]), share(x[1]), share(x[2])]
    return [sum(row) % Q for row in zip(*Y)]


INVERSE = 104491423396290281423421247963055991507
KAPPA = 6


# reduce accuracy
def truncate(value):
    b = add(value, [BASE ** (2 * PRECISION + 1), 0, 0])
    # only p0 knows the mask, re-conducts and gives it to p1 and p2
    mask = random.randrange(Q) % BASE ** (PRECISION + PRECISION_b + KAPPA)
    mask_low = mask % BASE ** PRECISION_b
    b_masked = reconstruct(add(b, [mask, 0, 0]))
    # get least significant bit
    b_masked_low = b_masked % BASE ** PRECISION_b
    b_low = sub(share(b_masked_low), share(mask_low))
    # delete bit
    c = sub(value, b_low)
    # delete other factors
    result = imul(c, INVERSE)
    return result


# addition
def add(x, y):
    return [(xi + yi) % Q for xi, yi in zip(x, y)]


# subtraction
def sub(x, y):
    return [(xi - yi) % Q for xi, yi in zip(x, y)]


# multiply fixed numbers
def imul(x, k):
    return [(xi * k) % Q for xi in x]


# multiplication
def mul(x, y):
    # local computation
    z0 = (x[0] * y[0] + x[0] * y[1] + x[1] * y[0]) % Q
    z1 = (x[1] * y[1] + x[1] * y[2] + x[2] * y[1]) % Q
    z2 = (x[2] * y[2] + x[2] * y[0] + x[0] * y[2]) % Q
    # re-share and assignment
    Z = [share(z0), share(z1), share(z2)]
    w = [sum(row) % Q for row in zip(*Z)]
    # reduce accuracy
    v = truncate(w)
    return v


# encryption
class SecureRational(object):

    def secure(secret):
        z = SecureRational()
        z.shares = share(encode(secret))
        return z

    def reveal(self):
        return decode(reconstruct(re_share(self.shares)))

    def __repr__(self):
        return "SecureRational(%f)" % self.reveal()

    def __add__(x, y):
        z = SecureRational()
        z.shares = add(x.shares, y.shares)
        return z

    def __sub__(x, y):
        z = SecureRational()
        z.shares = sub(x.shares, y.shares)
        return z

    def __mul__(x, y):
        z = SecureRational()
        z.shares = mul(x.shares, y.shares)
        return z

    def __pow__(x, e):
        z = SecureRational.secure(1)
        for _ in range(e):
            z = z * x
        return z


# secure operation for np list
secure = np.vectorize(lambda x: SecureRational.secure(x))
# decrypt np list
reveal = np.vectorize(lambda x: x.reveal())
