import os

def gen_run(desired_function, *args):
    try: result = desired_function(*args)
    except StopIteration:
        print('Error occured, ending run.')
        result = [None, None, None, None]
    finally: print('-- Closing model and netCDF file --')
    return result


def make_working_folder(foldername):
    try:  
        os.mkdir(foldername)
    except OSError:  
        print ("Fail: Couldn't create the output directory %s. It either failed or the directory already existed and files in it may be overwritten." % foldername)
    else:  
        print ("Success: Created the directory %s, and will save outputs there." % foldername)


def welcome_message():
    
    print(r' ')
    print(r' ')
    print(r' ')
    print(r'~~~~~~ Icy Worlds Ocean Model ~~~~~~')
 
    print(r'  _________________________________')
    print(r'      \  \   \______/  o<  /   /')
    print(r'       \  \_______________/   /')
    print(r'   >o   \____________________/')
    print(r'              >o       ')
    print(r'')    
    print(r'')
     
    
    
    