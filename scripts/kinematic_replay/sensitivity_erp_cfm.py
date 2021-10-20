#!/usr/bin/env python
""" Script to run kinematic replay. """
import os


def main():
    for erp in range(11):
        erp_val = float(erp/10)
        for cfm in range(11):
            cfm_val = float(cfm)
            command = f"run_kinematic_replay -erp {erp_val} -cfm {cfm_val}"
            print("Running: " + command)
            #os.system(command)
            
    

if __name__ == "__main__":
    """ Main """
    main()
