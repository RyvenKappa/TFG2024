
if __name__ == "__main__":
    import a
    import b
    from multiprocessing import Pipe
    cola = Pipe(duplex=False) #(Output,Input)
    proceso_a = a.Process_A(cola[1])
    proceso_b = b.Process_B(cola[0])
    proceso_a.start()
    proceso_b.start()
    proceso_a.join()
    proceso_b.join()