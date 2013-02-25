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
xn::Recorder recorder;

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
    
    //nRetVal = g_DepthGenerator0.Create(g_Context);
    //CHECK_RC(nRetVal, "creating depth");

  
    nRetVal = g_Context.EnumerateProductionTrees(XN_NODE_TYPE_DEPTH, NULL, depthList);
    xnPrintError(nRetVal, "Looking for depth generators..: ");

int i = 0;
    for (NodeInfoList::Iterator it = depthList.Begin();it != depthList.End(); ++it)
    {

        std::cout<<"i: "<<i<<std::endl;
            // Create the device node
            NodeInfo deviceInfo = *it;
            XnProductionNodeDescription nodeDesc = deviceInfo.GetDescription();
            cout << "Creating: " << nodeDesc.strName << " ,Vendor: " << nodeDesc.strVendor << ", Type: " << nodeDesc.Type << " Instance: " << deviceInfo.GetInstanceName() << endl;
            if(i == 0){
            	nRetVal = g_Context.CreateProductionTree(deviceInfo, g_DepthGenerator0);
            	CHECK_RC(nRetVal, "creating depth");
            }
            if(i == 1){
            	nRetVal = g_Context.CreateProductionTree(deviceInfo, g_DepthGenerator1);
            	CHECK_RC(nRetVal, "creating depth");
            }
            
            
            i++;
	}


	nRetVal = recorder.Create(g_Context);
	CHECK_RC(nRetVal, "Creating recorder");
	
	nRetVal = recorder.SetDestination(XN_RECORD_MEDIUM_FILE,"test.oni");
    CHECK_RC(nRetVal, "Setting up recorder file ");
	
	nRetVal = recorder.AddNodeToRecording(g_DepthGenerator0);
	CHECK_RC(nRetVal, "Adding node0 to recorder");
	nRetVal = recorder.AddNodeToRecording(g_DepthGenerator1);
	CHECK_RC(nRetVal, "Adding node1 to recorder");

	nRetVal = g_Context.StartGeneratingAll();
	CHECK_RC(nRetVal, "StartGenerating");


	while (!xnOSWasKeyboardHit()) {
		nRetVal = g_Context.WaitAndUpdateAll();
		CHECK_RC(nRetVal, "Wait'n'update");
		
		nRetVal = recorder.Record();
		CHECK_RC(nRetVal, "Record");
		            
	}

	g_Context.Shutdown();
	return 0;
}
