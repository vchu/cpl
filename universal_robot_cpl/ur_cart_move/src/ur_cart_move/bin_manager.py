#! /usr/bin/python

import numpy as np
from collections import deque
import yaml
from threading import RLock

import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("ar_track_alvar")
roslib.load_manifest('tf')
roslib.load_manifest('robotiq_c_model_control')
roslib.load_manifest('hrl_geom')
roslib.load_manifest('pykdl_utils')
import rospy

import tf
from geometry_msgs.msg import PoseStamped, PoseArray
from actionlib import SimpleActionClient

from ar_track_alvar.msg import AlvarMarkers
from hrl_geom.pose_converter import PoseConv
from traj_planner import TrajPlanner, pose_offset, pose_interp
from spline_traj_executor import SplineTraj
from ur_cart_move.msg import SplineTrajAction, SplineTrajGoal
from ur_cart_move.ur_cart_move import ArmInterface, RAVEKinematics
from robotiq_c_model_control.robotiq_c_ctrl import RobotiqCGripper
from pykdl_utils.kdl_kinematics import create_kdl_kin

BIN_HEIGHT_DEFAULT = 0.1
TABLE_OFFSET_DEFAULT = -0.2
TABLE_CUTOFF_DEFAULT = 0.05

class ARTagManager(object):
    def __init__(self):
        self.bin_height = rospy.get_param("~bin_height", BIN_HEIGHT_DEFAULT)
        self.table_offset = rospy.get_param("~table_offset", TABLE_OFFSET_DEFAULT)
        self.table_cutoff = rospy.get_param("~table_cutoff", TABLE_CUTOFF_DEFAULT)
        self.filter_window = rospy.get_param("~filter_window", 5.)

        lifecam_kin = create_kdl_kin('base_link', 'lifecam1_optical_frame')
        self.camera_pose = lifecam_kin.forward([])

        self.ar_poses = {}
        self.ar_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, 
                                       self.ar_cb)
        self.lock = RLock()

    def ar_cb(self, msg):
        with self.lock:
            cur_time = rospy.get_time()
            for marker in msg.markers:
                marker.pose.header = marker.header
                if marker.id not in self.ar_poses:
                    self.ar_poses[marker.id] = deque()
                self.ar_poses[marker.id].append([cur_time, marker.pose])
            for mid in self.ar_poses:
                while cur_time - self.ar_poses[mid][0][0] > self.filter_window:
                    self.ar_poses[mid].popleft()
                    if len(self.ar_poses[mid]) == 0:
                        break

    def get_available_bins(self):
        with self.lock:
            bins = []
            for mid in self.ar_poses:
                if len(self.ar_poses[mid]) > 0:
                    bins.append(mid)
            return bins

    def clean_ar_pose(self, ar_pose):
        ar_pose_mat = self.camera_pose * PoseConv.to_homo_mat(ar_pose)
        ang = np.arctan2(ar_pose_mat[1,0], ar_pose_mat[0,0])
        return (ar_pose_mat[:3,3].T.A[0], 
                [0., 0., ang])

    def get_bin_pose(self, bin_id):
        with self.lock:
            if bin_id not in self.ar_poses:
                print "Bin ID %d not found!" % bin_id
                return None, None
            pos_list, rot_list = [], []
            for cur_time, pose in self.ar_poses[bin_id]:
                pos, rot = self.clean_ar_pose(pose)
                pos_list.append(pos)
                rot_list.append(rot)
            print bin_id, np.array(pos_list)
            med_pos, med_rot = np.median(pos_list,0), np.median(rot_list,0)
            is_table = bool(med_pos[2] < self.table_cutoff)
            med_pos[2] = is_table*self.table_offset+self.bin_height
            return (med_pos.tolist(), med_rot.tolist()), is_table

    def get_all_bin_poses(self):
        with self.lock:
            bin_data = {}
            for bin_id in self.get_available_bins():
                bin_pose, is_table = self.get_bin_pose(bin_id)
                bin_data[bin_id] = [bin_pose[0], bin_pose[1], is_table]
            return bin_data

class BinManager(object):
    def __init__(self, arm_prefix, ar_empty_locs):
        self.bin_height = rospy.get_param("~bin_height", BIN_HEIGHT_DEFAULT)
        self.table_offset = rospy.get_param("~table_offset", TABLE_OFFSET_DEFAULT)
        self.table_cutoff = rospy.get_param("~table_cutoff", TABLE_CUTOFF_DEFAULT)

        self.place_offset = rospy.get_param("~place_offset", 0.02)
        self.ar_offset = rospy.get_param("~ar_offset", 0.115)
        self.grasp_height = rospy.get_param("~grasp_height", 0.10)
        self.grasp_rot = rospy.get_param("~grasp_rot", 0.0)
        self.grasp_lift = rospy.get_param("~grasp_lift", 0.18)
        self.waypt_offset = rospy.get_param("~waypt_offset", 0.25)
        self.waypt_robot_min_dist = rospy.get_param("~waypt_robot_min_dist", -0.20)

        self.pregrasp_vel = rospy.get_param("~pregrasp_vel", 2.0*0.10)
        self.grasp_dur = rospy.get_param("~grasp_dur", 3.50/2.0)
        self.grasp_vel = rospy.get_param("~grasp_vel", 0.03*2.0)

        self.qd_max = [0.2]*6
        self.q_min = [-4.78, -2.4, 0.3, -3.8, -3.3, -2.*np.pi]
        self.q_max = [-1.2, -0.4, 2.7, -1.6, 0.3, 2.*np.pi]

        self.ar_empty_locs = ar_empty_locs
        sim_prefix = rospy.get_param("~sim_arm_prefix", "/sim2")
        tf_list = tf.TransformListener()
        self.arm_cmd = ArmInterface(timeout=0., topic_prefix=arm_prefix)
        self.arm_sim = ArmInterface(timeout=0., topic_prefix=sim_prefix)
        if arm_prefix not in ["", "/"]:
            self.gripper = None
        else:
            self.gripper = RobotiqCGripper()
        self.kin = RAVEKinematics()
        self.traj_plan = TrajPlanner(self.kin)

        self.ar_man = ARTagManager()
        self.pose_pub = rospy.Publisher('/test', PoseStamped)
        #self.move_bin = rospy.ServiceProxy('/move_bin', MoveBin)
        self.traj_as = SimpleActionClient('spline_traj_as', SplineTrajAction)

        #print 'Waiting on TF'
        #rospy.sleep(0.5)
        #tf_list.waitForTransform('/base_link', '/lifecam1_optical_frame', 
        #                         rospy.Time(), rospy.Duration(3.0))
        #while not rospy.is_shutdown():
        #    try:
        #        now = rospy.Time.now()
        #        tf_list.waitForTransform('/base_link', '/lifecam1_optical_frame', 
        #                                 now, rospy.Duration(3.0))
        #        pose = tf_list.lookupTransform('/base_link', '/lifecam1_optical_frame', 
        #                                       now)
        #        print 'c'
        #        break
        #    except:
        #        continue
        #self.camera_pose = PoseConv.to_homo_mat(pose)
        #print 'Got TF transform for webcam frame'

        print 'Waiting for trajectory AS...', 
        self.traj_as.wait_for_server()
        print 'found.'
        if self.gripper is not None:
            print 'Waiting for gripper...', 
            self.gripper.wait_for_connection()
            print 'found.'
        print 'Waiting for arms...', 
        if (not self.arm_cmd.wait_for_states(timeout=1.) or
            not self.arm_sim.wait_for_states(timeout=1.)):
            print 'Arms not connected!'
        print 'found.'
        self.arm_cmd.set_payload(pose=([0., -0.0086, 0.0353], [0.]*3), payload=0.89)

        #while not rospy.is_shutdown():
        #    try:
        #        rospy.wait_for_service('/move_bin', timeout=0.01)
        #        break
        #    except (rospy.ROSException):
        #        continue

    def create_bin_waypts(self, ar_grasp_id, ar_place_id):

        ar_grasp_tag_pose, grasp_is_table = self.ar_man.get_bin_pose(ar_grasp_id)
        place_offset = self.place_offset*is_place
        grasp_offset = PoseConv.to_homo_mat(
                [0., self.ar_offset, self.grasp_height],
                [0., np.pi/2, self.grasp_rot])
        grasp_pose = PoseConv.to_homo_mat(ar_grasp_tag_pose) * grasp_offset
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[2,3] += self.grasp_lift

        ar_place_tag_pose, place_is_table = self.ar_empty_locs[ar_grasp_id]
        place_offset = PoseConv.to_homo_mat(
                [0., self.ar_offset, self.grasp_height + self.place_offset],
                [0., np.pi/2, self.grasp_rot])
        place_pose = PoseConv.to_homo_mat(ar_place_tag_pose) * place_offset
        preplace_pose = place_pose.copy()
        preplace_pose[2,3] += self.grasp_lift

        mid_pts = []
        if np.logical_xor(place_is_table, grasp_is_table):
            waypt_line = [0., 1., self.waypt_offset]
            grasp_pt = [grasp_pose[0,3], grasp_pose[1,3], 1.]
            place_pt = [place_pose[0,3], place_pose[1,3], 1.]
            waypt_pt = np.cross(waypt_line, np.cross(grasp_pt, place_pt))
            waypt_pt /= waypt_pt[2]
            waypt_pt[0] = min(waypt_pt[0], self.waypt_robot_min_dist)
            waypt = pose_interp(pregrasp_pose, preplace_pose, 0.5)
            waypt[:3,3] = np.mat([waypt_pt[0], waypt_pt[1], 
                                  self.grasp_height+self.bin_height+self.grasp_lift]).T
            mid_pts.append(waypt)

        waypts = (  [pregrasp_pose, grasp_pose, pregrasp_pose] 
                  + mid_pts 
                  + [preplace_pose, place_pose, preplace_pose])
        if False:
            r = rospy.Rate(1)
            for pose in waypts:
                self.pose_pub.publish(PoseConv.to_pose_stamped_msg("/base_link", pose))
                r.sleep()
            print pregrasp_pose, grasp_pose, pregrasp_pose, mid_pts, preplace_pose, place_pose, preplace_pose

        # bin_pose
        # ar_pose[0,0], 
        # x =
        # z = is_table*self.table_offset
        # bin_grasp/bin_place are [x, y, r, is_table]
        # move to pregrasp, above the bin (traversal)
        # move down towards the bin
        # grasp the bin
        # lift up
        # move to preplace (traversal)
        # move down
        # release bin
        return pregrasp_pose, grasp_pose, mid_pts, preplace_pose, place_pose

    def plan_move_traj(self, pregrasp_pose, mid_pts, preplace_pose, place_pose):

        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move upward to pregrasp pose
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=pregrasp_pose, dur=self.grasp_dur,
                vel_i=0., vel_f=self.grasp_vel,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move upward to pregrasp pose failed'
            print pregrasp_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        # move to preplace
        last_pose = pregrasp_pose
        for mid_pose in mid_pts + [preplace_pose]: 
            #q_mid_pose = self.kin.inverse_rand_search(mid_pose, q_knots[-1],
            #                                          pos_tol=0.001, rot_tol=np.deg2rad(1.))
            q_mid_pose = self.kin.inverse(mid_pose, q_knots[-1],
                                          q_min=self.q_min, q_max=self.q_max)
            if q_mid_pose is None:
                print 'failed move to replace'
                print mid_pose
                print q_knots
                return None
            q_knots.append(q_mid_pose)
            dist = np.linalg.norm(mid_pose[:3,3]-last_pose[:3,3])
            t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)
            last_pose = mid_pose

        # move downward to place pose
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=place_pose, dur=self.grasp_dur,
                vel_i=self.grasp_vel, vel_f=0.,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move downward to place pose'
            print place_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        move_traj = SplineTraj.generate(t_knots, q_knots)
        return move_traj

    def plan_retreat_traj(self, preplace_pose, place_pose):
        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move upward to preplace pose
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=preplace_pose, dur=self.grasp_dur,
                vel_i=0., vel_f=self.grasp_vel,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move upward to preplace pose failed'
            print preplace_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        q_home = [-2.75203516454, -1.29936272152, 1.97292018645, 
                  -2.28456617769, -1.5054511996, -1.1]
        q_knots.append(q_home)
        x_home = self.kin.forward(q_home)
        dist = np.linalg.norm(x_home[:3,3]-preplace_pose[:3,3])
        t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)

        retreat_traj = SplineTraj.generate(t_knots, q_knots)
        return retreat_traj

    def plan_home_traj(self):
        q_home = [-2.75203516454, -1.29936272152, 1.97292018645, 
                  -2.28456617769, -1.5054511996, -1.1]
        q_init = self.arm_cmd.get_q()
        start_time = rospy.get_time()
        t_knots, q_knots = self.traj_plan.min_jerk_interp_q_vel(q_init, q_home, self.pregrasp_vel)
        print 'Planning time:', rospy.get_time() - start_time
        home_traj = SplineTraj.generate(t_knots, q_knots)
        print t_knots, q_knots
        return home_traj

    def plan_grasp_traj(self, pregrasp_pose, grasp_pose):

        q_init = self.arm_cmd.get_q()
        x_init = self.kin.forward(q_init)

        q_knots = [q_init]
        t_knots = [0.]

        # move to pregrasp pose
        #q_pregrasp = self.kin.inverse_rand_search(pregrasp_pose, q_knots[-1],
        #                                          pos_tol=0.001, rot_tol=np.deg2rad(1.))
        q_pregrasp = self.kin.inverse(pregrasp_pose, q_knots[-1],
                                      q_min=self.q_min, q_max=self.q_max)
        if q_pregrasp is None:
            print 'move to pregrasp pose failed'
            print pregrasp_pose
            print q_knots
            return None
        q_knots.append(q_pregrasp)
        dist = np.linalg.norm(pregrasp_pose[:3,3]-x_init[:3,3])
        t_knots.append(t_knots[-1] + dist / self.pregrasp_vel)

        # move downward to grasp pose
        start_time = rospy.get_time()
        t_knots_new, q_knots_new = self.traj_plan.min_jerk_interp_ik(
                q_init=q_knots[-1], x_goal=grasp_pose, dur=self.grasp_dur,
                vel_i=self.grasp_vel, vel_f=0.,
                qd_max=self.qd_max, q_min=self.q_min, q_max=self.q_max)
        if q_knots_new is None:
            print 'move downward to grasp pose failed'
            print grasp_pose
            print q_knots
            return None
        print 'Planning time:', rospy.get_time() - start_time, len(t_knots_new)
        q_knots.extend(q_knots_new[1:])
        t_knots.extend(t_knots[-1] + t_knots_new[1:])

        grasp_traj = SplineTraj.generate(t_knots, q_knots)
        return grasp_traj

    def execute_traj(self, traj):
        goal = SplineTrajGoal(traj=traj.to_trajectory_msg())
        self.arm_cmd.unlock_security_stop()
        self.traj_as.wait_for_server()
        self.traj_as.send_goal(goal)
        self.traj_as.wait_for_result()
        result = self.traj_as.get_result()
        rospy.loginfo("Trajectory result:" + str(result))
        return result.success, result.is_robot_running

    def move_bin(self, grasp_tag, place_tag):
        (pregrasp_pose, grasp_pose, mid_pts, 
         preplace_pose, place_pose) = self.create_bin_waypts(grasp_tag, place_tag)
        grasp_traj = self.plan_grasp_traj(pregrasp_pose, grasp_pose)
        if grasp_traj is None:
            return False
        success, is_robot_running = self.execute_traj(grasp_traj)
        if not success:
            rospy.loginfo('Failed on grasping bin')
            return False
        if self.gripper is not None:
            self.gripper.close(block=True)
        move_traj = self.plan_move_traj(pregrasp_pose, mid_pts, preplace_pose, place_pose)
        if move_traj is None:
            return False
        success, is_robot_running = self.execute_traj(move_traj)
        if not success:
            rospy.loginfo('Failed on moving bin')
            return False
        if self.gripper is not None:
            self.gripper.goto(0.042, 0., 0., block=True)
            rospy.sleep(0.5)
        retreat_traj = self.plan_retreat_traj(preplace_pose, place_pose)
        if retreat_traj is None:
            return False
        success, is_robot_running = self.execute_traj(retreat_traj)
        if not success:
            rospy.loginfo('Failed on retreating from bin')
            return False
        return True

    def do_thing(self):
        raw_input("Move to home")
        reset = True
        while not rospy.is_shutdown():
            if reset:
                reset = False
                home_traj = self.plan_home_traj()
                self.execute_traj(home_traj)
                if self.gripper is not None:
                    if self.gripper.is_reset():
                        self.gripper.reset()
                        self.gripper.activate()
            if self.gripper is not None:
                if self.gripper.get_pos() != 0.042:
                    self.gripper.goto(0.042, 0., 0., block=False)
            #raw_input("Ready")
            ar_tags = self.ar_man.get_available_bins()
            grasp_tag_num = ar_tags[np.random.randint(0,len(ar_tags))]
            place_tag_num = grasp_tag_num
            while place_tag_num == grasp_tag_num:
                place_tag_num = ar_tags[np.random.randint(0,len(ar_tags))]
            if not self.move_bin(grasp_tag_num, place_tag_num):
                reset = True
                print 'Failed moving bin from %d to %d' % (grasp_tag_num, place_tag_num)

def main():
    np.set_printoptions(precision=4)
    rospy.init_node("bin_manager")

    from optparse import OptionParser
    p = OptionParser()
    p.add_option('-f', '--file', dest="filename", default="bin_locs.yaml",
                 help="YAML file of bin locations.")
    p.add_option('-s', '--save', dest="is_save",
                 action="store_true", default=False,
                 help="Save ar tag locations to file.")
    p.add_option('-t', '--test', dest="is_test",
                 action="store_true", default=False,
                 help="Test robot in simulations moving bins around.")
    (opts, args) = p.parse_args()

    if opts.is_save:
        ar_man = ARTagManager()
        rospy.sleep(4.)
        bin_data = ar_man.get_all_bin_poses()
        f = file(opts.filename, 'w')
        for bid in bin_data:
            print bid, bin_data[bid]
        yaml.dump({'data':bin_data}, f)
        f.close()

        poses_pub = rospy.Publisher('/ar_pose_array', PoseArray)
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            poses = PoseArray()
            poses.header.stamp = rospy.Time.now()
            poses.header.frame_id = '/base_link'
            for bid in bin_data:
                poses.poses.append(PoseConv.to_pose_msg(bin_data[bid][:2]))
            poses_pub.publish(poses)
            r.sleep()
    elif opts.is_test:
        f = file(opts.filename, 'r')
        ar_empty_locs = yaml.load(f)
        f.close()
        arm_prefix = "/sim1"
        bm = BinManager(arm_prefix, ar_empty_locs)
        bm.do_thing()
    else:
        print 'h'

    
    #bm = BinManager()
    #bm.do_thing()
    #rospy.spin()

if __name__ == "__main__":
    if False:
        import cProfile
        cProfile.run('main()', 'bm_prof')
    else:
        main()