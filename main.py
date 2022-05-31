import time
import pyopencl as cl
import numpy



class PP:
    def __init__(self, path):
        self.memfag = cl.mem_flags
        self.context = cl.create_some_context(interactive=True)
        self.queue = cl.CommandQueue(self.context)
        self.code = "".join(open(path, 'r').readlines())
        self.program = cl.Program(self.context, self.code).build()

    def getQueue(self):
        return self.queue

    def getProgram(self):
        return self.program

    def getFlags(self):
        return self.memfag

    def getContext(self):
        return self.context


if __name__ == '__main__':
    start_time = time.time()
    p = PP("./pi.cl")

    n, G, L = numpy.int32(pow(2,20)), pow(2,7), pow(2,7)
    x = numpy.zeros((n), dtype=numpy.double)

    a = numpy.array(x, dtype=numpy.double)

    a_buf = cl.Buffer(p.getContext(), p.getFlags().READ_ONLY | p.getFlags().COPY_HOST_PTR, hostbuf=x)


    p.getProgram().picalc(p.getQueue(), (G,), (L,), a_buf, n).wait()
    c = numpy.empty_like(x)
    cl._enqueue_read_buffer(p.getQueue(), a_buf, a).wait()
    sum=sum(a)
    pi=sum/n
    print('pi = ', pi)
    print(time.time() - start_time)
