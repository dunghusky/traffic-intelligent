from multiprocessing import Process, cpu_count
import time
 
def counter(number):
    init_amount = 0
    while init_amount < number:
        init_amount += 1
 
def main():
    start = time.perf_counter()
    a = Process(target=counter, args=(250000000,))
    b = Process(target=counter, args=(250000000,))
    c = Process(target=counter, args=(250000000,))
    d = Process(target=counter, args=(250000000,))

    a.start()
    b.start()
    c.start()
    d.start()

    a.join()
    b.join()
    c.join()
    d.join()

    print("PROCESSING")
    end = time.perf_counter()
    print("done in: ", int(end - start), 'Seconds')
    print("Your CPU has:",cpu_count(),"cores")

if __name__ == '__main__':
    main()