#!/usr/bin/python
import roslib
roslib.load_manifest("cpl_collab_manip")
roslib.load_manifest("project_simulation")
import rospy
import yaml
from roslaunch.substitution_args import resolve_args
from std_msgs.msg import UInt8MultiArray
from project_simulation.msg import move_bin as move_bin_msg
from project_simulation.msg import endf_bin
from cpl_collab_manip.bin_manager import BinManager 

PUB_RATE = 30
#if task is taking place currently or not
DOING_TASK = False
bm = None

#callback for move-bin messages
def listen_task(task_msg):
    global DOING_TASK
    if DOING_TASK:
        print 'Doing another task. Send message again.'
        return
    else:
        do_task(task_msg)

def task_failure():
    global DOING_TASK
    DOING_TASK = False
    return

def task_completion():
    global DOING_TASK
    DOING_TASK = False
    return

def do_task(task_msg):
    global bm, DOING_TASK
    DOING_TASK = True
    to_move_bin = task_msg.bin_id
    move_near_human = task_msg.move_near_human
    slot_to_occupy = None
    
    #locate the bin
    bin_found = False
    avail_bin_list = bm.ar_man.get_available_bins()
    print avail_bin_list
    for bin_id in avail_bin_list:
        if bin_id == to_move_bin:
            bin_found = True
            break
    if not bin_found:        
        print 'Could not complete task the task of moving bin-id: ', to_move_bin, ' because bin was not found.'
        task_failure()
    else:
        bins_in_slots, missing_bins = bm.ar_man.get_bin_slot_states()
        
        if move_near_human:
            #check if bin already in workspace
            bins_near_human = bm.ar_man.get_filled_slots(True)
            for bin_near_hum in bins_near_human:
                if bin_near_hum == to_move_bin:
                    print 'Bin %d already in workspace' % (to_move_bin)
                    task_completion()
                    return
            #check if there's an empty slot in the workspace
            cur_bin_pose, on_table = bm.ar_man.get_bin_pose(to_move_bin) 
            slot_to_occupy = bm.ar_man.get_empty_slot(True, cur_bin_pose[1]) 
            if  slot_to_occupy < 0:
                print 'No empty slot in workspace. Skipping task.'
                task_failure()
                return
        else:
            #check if bin already outside workspace
            bins_away_human = bm.ar_man.get_filled_slots(False)
            for bin_away_hum in bins_away_human:
                if bin_away_hum == to_move_bin:
                    print 'Bin %d already outside workspace' % (to_move_bin)
                    task_completion()
                    return
            #check if there's an empty slot out of the workspace
            cur_bin_pose, on_table = bm.ar_man.get_bin_pose(to_move_bin) 
            slot_to_occupy = bm.ar_man.get_empty_slot(False, cur_bin_pose[1]) 
            if  slot_to_occupy < 0:
                print 'No empty slot outside workspace. Skipping task.'
                task_failure()
                return
        print '*'*30, 'MOVE OCCURING'
        print 'bins in workspace-', bm.ar_man.get_filled_slots(True)
        print 'bins in slots- ', bins_in_slots
        print 'slot to move to-', slot_to_occupy
        if bm.move_bin(to_move_bin, slot_to_occupy):
            print 'Task Completed'
            task_completion()
            return            
        else:
            print 'Failed to perform task'
            print 'Resetting system..'
            bm.system_reset()
            return
    

#---------MAIN---------
def main():
    global bm
    
    rospy.init_node("robot_mover")
    print 'Starting robot bin-mover...'
    arm_prefix = ""
    f_name = "$(find cpl_collab_manip)/config/bin_locs.yaml"
    f = file(resolve_args(f_name))
    bin_slots = yaml.load(f)['data']
    bm = BinManager(arm_prefix, bin_slots)
    
    #reset bin manager system first
    print 'Commence system reset'
    bm.system_reset()

    ''' Debugger Y'all
    print '*'*50
    print 'available bins = ', bm.ar_man.get_available_bins()
    print '*'*50
    print 'pose for bin 2', bm.ar_man.get_bin_pose(2)
    print '*'*50
    print 'get slot IDs', bm.ar_man.get_slot_ids()
    print '*'*50
    #print 'bin_slots', bm.ar_man.get_bin_slot
    print 'bin slot states', bm.ar_man.get_bin_slot_states()
    print '*'*50
    print 'all empty slots', bm.ar_man.get_empty_slots()
    print '*'*50
    print 'get the ones near mr.human', bm.ar_man.get_empty_slots(True)
    print '*'*50
    print 'get the ones far from  mr.human', bm.ar_man.get_empty_slots(True)
    print '*'*50'''

    #setup listening
    task_listen_sub = rospy.Subscriber('move_bin', move_bin_msg, listen_task, queue_size=1)
    #publish work-bins
    pub_work_bins = rospy.Publisher('workspace_bins', UInt8MultiArray)
    #publish endfactor
    pub_endf = rospy.Publisher('endfactor_bin', endf_bin)

    bins_wrk_msg = UInt8MultiArray()
    no_of_work_bins = len(bm.ar_man.human_slots)
    endf_msg = endf_bin()

    #rospy.spin()
    loop_rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        bins_near_human = bm.ar_man.get_filled_slots(True)
        for i in range(len(bins_near_human)):
            if bins_near_human[i] < 0:
                bins_near_human[i] = 0
        for i in range(no_of_work_bins-len(bins_near_human)):
            bins_near_human.append(0)
        bins_wrk_msg.data = bins_near_human

        pub_work_bins.publish(bins_wrk_msg)
        
        #debug
        #print bins_wrk_msg
        
        endf_msg.performing_task.data = DOING_TASK
        pub_endf.publish(endf_msg)

        loop_rate.sleep()

if __name__=='__main__':
    main()
