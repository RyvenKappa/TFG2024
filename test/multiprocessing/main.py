
if __name__ == "__main__":
    import a
    import b
    from multiprocessing import Queue
    cola = Queue()
    proceso_a = a.Process_A(cola)
    proceso_b = b.Process_B(cola)
    proceso_a.start()
    proceso_b.start()
    proceso_a.join()
    proceso_b.join()