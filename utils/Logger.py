

# Note: This code is based on Singleton Design pattern where object instance creation will happen once for multiple instance 
#       and same is case for object deletion.

class Logger:
    logger = None       # This would work as a static variable and shared among all the instances of Logger classes.
    count  = 0
    def __init__(self, log_path, log_level=None):
        
        if log_level is None:
            self.log_level = ""
        else:
            self.log_level = "[{}] ".format(log_level)    

        if Logger.count == 0:
            # Only once logger should be initialized with given path.
            Logger.logger = open(log_path, 'w', encoding='utf-8')
            
        Logger.count += 1

        self.log_ctr = 0


    def __call__(self, message):
        print(self.log_level + message)
        Logger.logger.writelines(self.log_level + message + "\n")
        self.log_ctr += 1
        if(self.log_ctr % 10 == 0):
            # Flush data into logger at every 10 writes.
            Logger.logger.flush()


    def __del__(self):
        Logger.count -= 1
        if Logger.count == 0:
            # Closing the file just for the safety purpose, when all the instances are deleted.
            Logger.logger.close()
            print("Logger file closed")
            
            