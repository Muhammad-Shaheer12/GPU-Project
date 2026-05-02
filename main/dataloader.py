import json
import numpy as np
import threading
from queue import Queue

class FastYelpDataLoader:
    def __init__(self, json_path, tokenizer, batch_size=2048, max_length=128, prefetch_batches=100):
        self.json_path = json_path
        self.tokenizer = tokenizer
        
        # GPU constraints (VRAM limit)
        self.batch_size = batch_size
        self.max_length = max_length
        
        # CPU/RAM constraints (System RAM utilization)
        # prefetch_batches=100 means we hold 100 fully prepared batches in system RAM
        self.queue = Queue(maxsize=prefetch_batches) 
        
        # A special marker to tell the GPU when the dataset is finished
        self._STOP_MARKER = object() 

    def _data_preparation_worker(self):
        """This runs in the background. It reads disk, tokenizes, and fills the queue."""
        X_batch_list = []
        y_batch_list = []

        with open(self.json_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    text = review.get('text', '')
                    rating = int(review.get('stars', 3))
                    
                    encoded_text = self.tokenizer.encode(text, self.max_length)
                    
                    X_batch_list.append(encoded_text)
                    y_batch_list.append(rating)

                    if len(X_batch_list) == self.batch_size:
                        X_batch = np.array(X_batch_list, dtype=np.int32)
                        y_batch = np.array(y_batch_list, dtype=np.int32)
                        
                        # Shove it into system RAM. 
                        # If the queue is full (100 batches), this thread automatically pauses 
                        # until the GPU takes one out.
                        self.queue.put((X_batch, y_batch))
                        
                        X_batch_list = []
                        y_batch_list = []
                        
                except json.JSONDecodeError:
                    continue
                    
        # When the file is completely read, put the stop marker in the queue
        self.queue.put(self._STOP_MARKER)

    def get_batches(self):
        """The main thread calls this to instantly get data for the GPU."""
        
        # 1. Start the background worker thread
        worker = threading.Thread(target=self._data_preparation_worker)
        worker.daemon = True # Ensures thread dies if the main script crashes
        worker.start()

        # 2. Continually pop prepared batches from the queue
        while True:
            batch = self.queue.get()
            
            # If we hit the marker, the file is done
            if batch is self._STOP_MARKER:
                break
                
            yield batch