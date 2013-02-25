// openni_tracker.cpp

#include <string>
#include <vector>
#include <ros/ros.h>
#include <ros/package.h>

#include <XnOpenNI.h>
#include <XnCodecIDs.h>
#include <XnCppWrapper.h>

using std::string;
using namespace std;
using namespace xn;

xn::Context        g_Context;
xn::DepthGenerator g_DepthGenerator0;
xn::DepthGenerator g_DepthGenerator1;
xn::UserGenerator g_UserGenerator;
xn::Recorder recorder;


XnBool g_bNeedPose   = FALSE;
XnChar g_strPose[20] = "";


void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie) {
        ROS_INFO("New User %d", nId);

        if (g_bNeedPose)
                g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
        else
                g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie) {
        ROS_INFO("Lost user %d", nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& capability, XnUserID nId, void* pCookie) {
        ROS_INFO("Calibration started for user %d", nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationEnd(xn::SkeletonCapability& capability, XnUserID nId, XnBool bSuccess, void* pCookie) {
        if (bSuccess) {
                ROS_INFO("Calibration complete, start tracking user %d", nId);
                g_UserGenerator.GetSkeletonCap().StartTracking(nId);
        }
        else {
                ROS_INFO("Calibration failed for user %d", nId);
                if (g_bNeedPose)
                        g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
                else
                        g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
        }
}

void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& capability, XnChar const* strPose, XnUserID nId, void* pCookie) {
    ROS_INFO("Pose %s detected for user %d", strPose, nId);
    g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
    g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

#define CHECK_RC(nRetVal, what)										\
	if (nRetVal != XN_STATUS_OK)									\
	{																\
		ROS_ERROR("%s failed: %s", what, xnGetStatusString(nRetVal));\
		return nRetVal;												\
	}

int main(int argc, char **argv) {
    ros::init(argc, argv, "openni_tracker_double");
    XnStatus nRetVal = XN_STATUS_OK;
    
    NodeInfoList userList;
    NodeInfoList depthList;
    
    nRetVal = g_Context.Init();
    CHECK_RC(nRetVal, "Init context");

	xn::Player player;

	nRetVal = g_Context.OpenFileRecording("test.oni", player);
	CHECK_RC(nRetVal, "Open recording");
	
	player.SetRepeat(false);
	
	
	
	
	//nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator0);
	//CHECK_RC(nRetVal, "Exisiting depth node");
	


    
    nRetVal = g_Context.EnumerateProductionTrees(XN_NODE_TYPE_DEPTH, NULL, depthList);
    xnPrintError(nRetVal, "Looking for depth generators..: ");

int i = 0;
int targetDevice = 1;
    for (NodeInfoList::Iterator it = depthList.Begin();it != depthList.End(); ++it)
    {

        std::cout<<"i: "<<i<<std::endl;
            // Create the device node
            NodeInfo deviceInfo = *it;
            XnProductionNodeDescription nodeDesc = deviceInfo.GetDescription();
            cout << "Creating: " << nodeDesc.strName << " ,Vendor: " << nodeDesc.strVendor << ", Type: " << nodeDesc.Type << " Instance: " << deviceInfo.GetInstanceName() << endl;
            if(i == targetDevice){
            	nRetVal = g_Context.CreateProductionTree(deviceInfo, g_DepthGenerator0);
            	xnPrintError(nRetVal, "Creating depthGen.: ");
            	
            }
            //if(i == targetDevice){
            //	nRetVal = g_Context.CreateProductionTree(deviceInfo, g_DepthGenerator1);
            //	xnPrintError(nRetVal, "Creating depthGen.: ");
            //}
            
            
            i++;
	}
	XnUInt32 uFrames;
	player.GetNumFrames(g_DepthGenerator0.GetName(), uFrames);
	
	cout << uFrames << endl;
	
	nRetVal = g_UserGenerator.Create(g_Context);
	CHECK_RC(nRetVal, "Creating user gen.");


if(g_UserGenerator.IsValid())
{
	cout << "User generator is working.." << endl;

	XnCallbackHandle h1, h2, h3;
	g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, h1);
	g_UserGenerator.GetPoseDetectionCap().RegisterToPoseCallbacks(UserPose_PoseDetected, NULL, NULL, h2);
	g_UserGenerator.GetSkeletonCap().RegisterCalibrationCallbacks(UserCalibration_CalibrationStart, 		      UserCalibration_CalibrationEnd, NULL, h3);
	//genUser.GetSkeletonCap().SetSmoothing(1.0);
	
	// Set the profile
	g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
}
if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration()) {

cout << "Requires pose" << endl;
	XnCallbackHandle hPoseCallbacks;
	g_UserGenerator.GetPoseDetectionCap().RegisterToPoseCallbacks(UserPose_PoseDetected, NULL, NULL, hPoseCallbacks);

	g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
}
cout << "User generator is working11.." << endl;
	XnSkeletonJointPosition joint_position;
    
cout << "User generator is working22." << endl;

	XnUInt64 timestamp;
	XnUInt32 nFrame;
	

	nRetVal = g_Context.StartGeneratingAll();
	CHECK_RC(nRetVal, "Start generating");

	while (!xnOSWasKeyboardHit()) {
	
		nRetVal = g_Context.WaitAndUpdateAll();
		CHECK_RC(nRetVal, "Wait'n'update");
		
		nRetVal = player.TellTimestamp(timestamp);
		CHECK_RC(nRetVal, "Timestamp: ");
		
		nRetVal = player.TellFrame(g_DepthGenerator0.GetName(), nFrame);
		
		
		cout << "Time: " << timestamp << endl;
		cout << "Frame: " << nFrame << endl;
		
				
		g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(1, XN_SKEL_HEAD, joint_position);
		
		cout << "Head position z: " << joint_position.position.Z << endl;
				

		            
	}

	g_Context.Shutdown();
	return 0;
}
