import torch
import torch.tensor
from multiprocessing.pool import ThreadPool
import threading
import queue
import six
import time

class TensorLoader():
    def __init__( self, np_seq ):
        self.np_seq = np_seq 
    def __iter__( self ):
        return self
    #def __next__(self):
    #    tup = self.np_seq.next()
    #    return tuple(map(torch.from_numpy, tup) )
    def __next__( self ):
        tup = self.np_seq.next()
        retx = torch.from_numpy( tup[0] )
        if len(tup) <= 1:
            return (retx)
        # Torch needs LongTensor
        rety = torch.from_numpy( tup[1].astype(int))
        # print(rety)
        return (retx, rety)
    def len(self):
        return self.np_seq.len()
    def __len__(self):
        return len(self.np_seq)
    
def to_loader( np_seq ):
    return TensorLoader( np_seq )

class TensorLoader():
    def __init__( self, np_seq ):
        self.np_seq = np_seq 
    def __iter__( self ):
        return self
    #def __next__(self):
    #    tup = self.np_seq.next()
    #    return tuple(map(torch.from_numpy, tup) )
    def __next__( self ):
        tup = self.np_seq.next()
        retx = torch.from_numpy( tup[0] )
        if len(tup) <= 1:
            return (retx)
        # Torch needs LongTensor
        rety = torch.from_numpy( tup[1].astype(int))
        # print(rety)
        return (retx, rety)
    def len(self):
        return self.np_seq.len()
    def __len__(self):
        return len(self.np_seq)
    
class SequenceEnqueuer():
    def __init__(self, generator,
                 allow_exception = False, 
                 wait_time=0.001, 
                 number_warning = 1000000, 
                 monitor_queue = False ):
        self.number_warning = number_warning
        self.wait_time = wait_time
        self._generator = to_loader( generator )
        self._threads = []
        self._stop_event = None
        self._manager = None
        self.queue = None
        self.end_reached = False
        self.allow_exception = allow_exception
        self.monitor_queue = monitor_queue
        self.start_time = time.time()
        
    def __next__( self ):
        while self.is_running():
            if not self.queue.empty():
                success, value = self.queue.get()
                # Rethrow any exceptions found in the queue
                if not success:
                    six.reraise(value.__class__, value, value.__traceback__)
                # Yield regular values
                if value is not None:
                    if self.monitor_queue:
                        cur_queue_size = self.queue.qsize()
                        print ("Obtain one minibatch, current queue size == %d" % cur_queue_size )
                    return value
            else:
                all_finished = self.end_reached # all([not thread.is_alive() for thread in self._threads])
                if all_finished and self.queue.empty():
                    # stop generator task
                    # print ("End of epoch reached, raise StopIteration" )
                    self._stop_event.set()
                    raise StopIteration
                else:
                    cur = time.time()
                    elapse = cur - self.start_time
                    if elapse > 10.0: 
                        if self.number_warning > 0:
                            self.number_warning -= 1
                            self.start_time = cur
                            print( "Data queue drained, please consider to increase the number of worker thread" )
                    time.sleep(self.wait_time)

        # Make sure to rethrow the first exception in the queue, if any
        while not self.queue.empty():
            success, value = self.queue.get()
            if not success:
                six.reraise(value.__class__, value, value.__traceback__)
            return value
        
        # print ("End of epoch reached, raise StopIteration" )
        raise StopIteration
        
    def len(self):
        return self._generator.len()
    def __len__(self):
        return len(self._generator) 

    def _data_generator_task(self):
        while not self._stop_event.is_set():
            with self.genlock:
                try:
                    if (self.queue is not None and
                            self.queue.qsize() < self.max_queue_size):
                        # On all OSes, avoid **SYSTEMATIC** error
                        # in multithreading mode:
                        # `ValueError: generator already executing`
                        # => Serialize calls to
                        # infinite iterator/generator's next() function
                        generator_output = next(self._generator)
                        self.queue.put((True, generator_output))
                    else:
                        time.sleep(self.wait_time)
                except StopIteration:
                    # print( "Stop Iterateration encountered" )
                    self._stop_event.set()
                    self.end_reached = True
                    break
                except Exception as e:
                    # Can't pickle tracebacks.
                    # As a compromise, print the traceback and pickle None instead.
                    if not self.allow_exception:
                        if not hasattr(e, '__traceback__'):
                            setattr(e, '__traceback__', sys.exc_info()[2])
                        # print( "Exception occurred %s " % e )
                        self.queue.put((False, e))
                        self._stop_event.set()
                        self.end_reached = True
                    else:
                        # Otherwise, swallow exception
                        ()
                    break

    def start(self, workers = 1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """
        try:
            self.end_reached = False
            self.max_queue_size = max_queue_size
            self.genlock = threading.Lock()
            self.queue = queue.Queue(maxsize=max_queue_size)
            self._stop_event = threading.Event()

            for _ in range(workers):
                thread = threading.Thread(target=self._data_generator_task)
                self._threads.append(thread)
                thread.start()
        except Exception as e:
            # print( "Data generator thread stops for exception %s" % e )
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        # print( "SequenceEnqueuer stop is called" )
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout)

        self._threads = []
        self._stop_event = None
        self.queue = None
    
    
class TensorEnqueuer():
    """Builds a queue to feed tensor, 
    """
    def __init__(self, generator,
                 max_queue_size = 10,
                 allow_exception = False, 
                 wait_time = 0.001
                 ) : 
        self.max_queue_size = max_queue_size
        self.allow_exception = allow_exception
        self.wait_time = wait_time
        self._generator = generator
    def len(self):
        return self._generator.len()
    def __len__(self):
        return len(self._generator)
    def __iter__(self):
        # print ("Iterator on TensorEnqueue called" )
        # This allow multiple loop to start over the current TensorEnqueuer
        sequence = SequenceEnqueuer( self._generator, allow_exception = self.allow_exception, wait_time = self.wait_time ) 
        sequence.start( max_queue_size = self.max_queue_size )
        return sequence
    
    
def to_queued_loader( np_seq, max_queue_size=20, wait_time=0.001, allow_exception = False ):
    return TensorEnqueuer( np_seq, max_queue_size = max_queue_size, wait_time = wait_time, allow_exception = allow_exception )

class DummyTensorLoader():
    def __init__( self, np_seq ):
        self.np_seq = np_seq 
        self.tup = self.np_seq.next()
        self.retx = torch.from_numpy( self.tup[0] )
        if len(self.tup) > 1:
            self.rety = torch.from_numpy( self.tup[1].astype(int))
    def __iter__( self ):
        return self
    #def __next__(self):
    #    tup = self.np_seq.next()
    #    return tuple(map(torch.from_numpy, tup) )
    def __next__( self ):
        if len(self.tup) <= 1:
            return (self.retx)
        return (self.retx, self.rety)
    def len(self):
        return self.np_seq.len()
    def __len__(self):
        return len(self.np_seq)
    
def to_dummy_loader( np_seq ):
    return DummyTensorLoader( np_seq )

