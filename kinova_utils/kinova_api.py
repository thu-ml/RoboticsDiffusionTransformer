import time
import sys, os
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import kinova_utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def Move_to_zero_position(base):
    # 使机械臂处于直立姿态
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Move arm to ready position
    constrained_joint_angles = Base_pb2.ConstrainedJointAngles()

    actuator_count = base.GetActuatorCount().count
    angles = [0.0] * actuator_count

    for joint_id in range(len(angles)):
        joint_angle = constrained_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.PlayJointTrajectory(constrained_joint_angles)

    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        return True
    else:
        return False

def Send_joint_speeds(base, target_joint_speeds = [0]*7): # 控制机械臂各关节速度，单位为角度每秒
    joint_speeds = Base_pb2.JointSpeeds()
    actuator_count = base.GetActuatorCount().count

    for i, speed in enumerate(target_joint_speeds):
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i 
            joint_speed.value = target_joint_speeds[i]
            joint_speed.duration = 0
            i = i + 1
    
    base.SendJointSpeedsCommand(joint_speeds)
    # 需要time.sleep()

def Move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        return True
    else:
        return False

def Move_to_retract_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Retract":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        return True
    else:
        return False

def Send_joint_angles(base, target_joint_angles = [0]*7): # 关节角度控制，单位为角度
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = target_joint_angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    base.ExecuteAction(action)

    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        return True
    else:
        return False

def Send_joint_angles_no_wait(base, target_joint_angles = [0]*7): # 关节角度控制，单位为角度
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = target_joint_angles[joint_id]

    base.ExecuteAction(action)
    time.sleep(0.1)

def Get_cartesian_pose(base_cyclic):
    feedback = base_cyclic.RefreshFeedback()

    pose_x = feedback.base.tool_pose_x
    pose_y = feedback.base.tool_pose_y
    pose_z = feedback.base.tool_pose_z
    theta_x = feedback.base.tool_pose_theta_x
    theta_y = feedback.base.tool_pose_theta_y
    theta_z = feedback.base.tool_pose_theta_z

    return [pose_x, pose_y, pose_z, theta_x, theta_y, theta_z]

def Send_relative_cartesian_pose(base, base_cyclic, relative_pose):
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + relative_pose[0]             # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y + relative_pose[1]             # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z + relative_pose[2]             # (meters)
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + relative_pose[3] # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y + relative_pose[4] # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z + relative_pose[5] # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteAction(action)

    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        return True
    else:
        return False

def Send_abosolute_cartesian_pose(base, base_cyclic, abosolute_pose):
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = abosolute_pose[0]              # (meters)
    cartesian_pose.y = abosolute_pose[1]              # (meters)
    cartesian_pose.z = abosolute_pose[2]              # (meters)
    cartesian_pose.theta_x = abosolute_pose[3]        # (degrees)
    cartesian_pose.theta_y = abosolute_pose[4]        # (degrees)
    cartesian_pose.theta_z = abosolute_pose[5]        # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteAction(action)

    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        return True
    else:
        return False

def Send_twist_command(base, command): # 末端线速度和角速度控制
    twist_command = Base_pb2.TwistCommand()

    twist_command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    twist_command.duration = 0

    # 使用前请先检查xyz，rpy对应关系
    twist = twist_command.twist
    twist.linear_x = command[0]
    twist.linear_y = command[1]
    twist.linear_z = command[2]
    twist.angular_x = command[3]
    twist.angular_y = command[4]
    twist.angular_z = command[5]

    base.SendTwistCommand(twist_command)

    # 必需
    time.sleep(10)
    return True

def Close_gripper(base):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        position = 1.0
        finger.finger_identifier = 1
        finger.value = position
        base.SendGripperCommand(gripper_command)
        time.sleep(0.7)

def Open_gripper(base):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        position = 0.0
        finger.finger_identifier = 1
        finger.value = position
        base.SendGripperCommand(gripper_command)
        time.sleep(0.7)

def Send_gripper_position(base, x): # 0 <= x <=1
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    # Close the gripper with position increments
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = x
    base.SendGripperCommand(gripper_command)
    time.sleep(0.1)

def Get_joint_positions(base, base_cyclic):
    feedback = base_cyclic.RefreshFeedback()
    actuator_count = base.GetActuatorCount().count

    joints = []

    for i in range(actuator_count):
        joints.append(feedback.actuators[i].position)

    return joints

def populateCartesianCoordinate(waypointInformation):
    
    waypoint = Base_pb2.CartesianWaypoint()  
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    waypoint.blending_radius = waypointInformation[3]
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6] 
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    
    return waypoint

def Send_Cartesian_waypoints(base, base_cyclic, waypointsTuple):
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    product = base.GetProductConfiguration()

    if(product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L53):
        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = True

        index = 0
        for waypointDefinition in waypointsTuple:
            waypoint = waypoints.waypoints.add()
            waypoint.name = "waypoint_" + str(index)   
            waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypointDefinition))
            index = index + 1

    # Verify validity of waypoints
    result = base.ValidateWaypointList(waypoints);
    if(len(result.trajectory_error_report.trajectory_error_elements) == 0):
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(   check_for_end_or_abort(e),
                                                                Base_pb2.NotificationOptions())
        base.ExecuteWaypointTrajectory(waypoints)
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian trajectory with optimization completed ")
        else:
            print("Timeout on action notification wait for non-optimized trajectory")

        return finished
        
    else:
        print("Error found in trajectory") 
        result.trajectory_error_report.PrintDebugString(); 

def connection(parser):
    args = kinova_utilities.parseConnectionArguments(parser)
    cnnt = kinova_utilities.DeviceConnection(args.ip, port=kinova_utilities.DeviceConnection.TCP_PORT, credentials=(args.username, args.password))
    router = cnnt.__enter__()
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)
    return cnnt, base, base_cyclic

def disconnect(cnnt):
    cnnt.__exit__(None, None, None)
    
