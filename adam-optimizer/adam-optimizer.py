import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    
    m=beta1*m+(1-beta1)*grad
    v=beta2*v+(1-beta2)*grad**2

    m_correc = m/(1-beta1**t)
    v_correc = v/(1-beta2**t)

    param = param - lr * m_correc/(np.sqrt(v_correc)+eps)

    return (param, m, v)