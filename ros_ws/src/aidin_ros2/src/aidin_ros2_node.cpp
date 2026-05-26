#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <lxros/lxros.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp> // geometry_msgs::msg::Twist
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <aidin_ros2/ExpFilter.h>

#define NUM_FLOATS 13
#define BYTE_SIZE 52


geometry_msgs::msg::WrenchStamped wrench;
geometry_msgs::msg::WrenchStamped wrench_raw;

geometry_msgs::msg::WrenchStamped bias;
std::vector<ExponentialFilter> filters;


struct FT_Sensor{
    float F[3];
    float T[3];
    float acc[3];
    float gyro[3];
    float temp;
};


bool bias_sensor(std_srvs::srv::Empty::Request  &req,
         std_srvs::srv::Empty::Response &res)
{
    printf("Sensor Biased\n");
    bias=wrench_raw;
    printf("%.3f,%.3f,%.3f%.3f,%.3f,%.3f\n",bias.wrench.force.x,bias.wrench.force.y,bias.wrench.force.z,bias.wrench.torque.x,bias.wrench.torque.y,bias.wrench.torque.z);
    return true;
}


float bytes_to_double(unsigned char buffer[4]){
    float val;
    for (int j=0;j<4;j++){
        ((unsigned char*)&val)[j] = buffer[3-j];
    }        
    //printf("V: %f\n",val);
    return val;
}


void buffer_to_wrench_and_acc(unsigned char buffer[BYTE_SIZE],geometry_msgs::msg::Wrench *ft,geometry_msgs::msg::Vector3 *acc,geometry_msgs::msg::Vector3 *gyro, double *temp){
    wrench_raw.wrench.force.x=bytes_to_double(buffer+0);
    wrench_raw.wrench.force.y=bytes_to_double(buffer+4);
    wrench_raw.wrench.force.z=bytes_to_double(buffer+8);
    wrench_raw.wrench.torque.x=bytes_to_double(buffer+12);
    wrench_raw.wrench.torque.y=bytes_to_double(buffer+16);
    wrench_raw.wrench.torque.z=bytes_to_double(buffer+20);
    // Apply bias
    ft->force.x=wrench_raw.wrench.force.x-bias.wrench.force.x;
    ft->force.y=wrench_raw.wrench.force.y-bias.wrench.force.y;
    ft->force.z=wrench_raw.wrench.force.z-bias.wrench.force.z;
    ft->torque.x=wrench_raw.wrench.torque.x-bias.wrench.torque.x;
    ft->torque.y=wrench_raw.wrench.torque.y-bias.wrench.torque.y;
    ft->torque.z=wrench_raw.wrench.torque.z-bias.wrench.torque.z;   

    acc->x=bytes_to_double(buffer+24);
    acc->y=bytes_to_double(buffer+28);
    acc->z=bytes_to_double(buffer+32);

    gyro->x=bytes_to_double(buffer+36);
    gyro->y=bytes_to_double(buffer+40);
    gyro->z=bytes_to_double(buffer+44);

    *temp=bytes_to_double(buffer+48);

}



geometry_msgs::msg::Wrench filterWrench(geometry_msgs::msg::Wrench in){
    geometry_msgs::msg::Wrench out;
    out.force.x=filters[0].apply(in.force.x);
    out.force.y=filters[1].apply(in.force.y);
    out.force.z=filters[2].apply(in.force.z);
    out.torque.x=filters[3].apply(in.torque.x);
    out.torque.y=filters[4].apply(in.torque.y);
    out.torque.z=filters[5].apply(in.torque.z);
    return out;
}


void buffer_to_struct(unsigned char buffer[BYTE_SIZE],struct FT_Sensor *ft){
    unsigned char buffer_le[BYTE_SIZE];    
    for (int i=0;i<NUM_FLOATS;i++){
        
    }
    memcpy(ft,buffer_le,BYTE_SIZE);
}

int main(int argc, char **argv) {

    for (int i=0;i<6;i++) filters.push_back(ExponentialFilter(0.1f));

    lxros::init(argc, argv);

    lxros::LxNode node("haply_node");


    std::string ft_sensor_ip;
    int ft_sensor_port;

    ft_sensor_ip=node.get_param<std::string>("ft_sensor_ip", "192.168.1.199");
    ft_sensor_port=node.get_param<int>("ft_sensor_port",8890);

    lxros::Publisher<geometry_msgs::msg::WrenchStamped> sensor_pub =
        node.pub<geometry_msgs::msg::WrenchStamped>("ft_sensor", 100);
    lxros::Publisher<geometry_msgs::msg::WrenchStamped> sensor_filt_pub =
        node.pub<geometry_msgs::msg::WrenchStamped>("ft_sensor_filtered", 100);
    lxros::Publisher<sensor_msgs::msg::Imu> imu_pub =
        node.pub<sensor_msgs::msg::Imu>("ft_imu", 100);

    

    lxros::ServiceServer bias_service = node.service_server<std_srvs::srv::Empty>("ft_sensor_bias", bias_sensor);
    

    // Step 1: Create the socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }

    // Step 2: Define the server address
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(ft_sensor_port);
    serverAddr.sin_addr.s_addr = inet_addr(ft_sensor_ip.c_str()); // Send to localhost, change to your server IP

    // Step 3: Send three bytes [0x00, 0x03, 0x02]
    unsigned char dataToSend[3] = {0x00, 0x03, 0x02};
    ssize_t bytesSent = sendto(sockfd, dataToSend, sizeof(dataToSend), 0, 
                                (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bytesSent < 0) {
        std::cerr << "Error sending data" << std::endl;
        close(sockfd);
        return -1;
    }
    std::cout << "Sent 3 bytes: [0x00, 0x03, 0x02]" << std::endl;

    // Step 4: Start receiving data (52 bytes total)
    unsigned char buffer[BYTE_SIZE]; // Buffer to store 52 incoming bytes
    struct sockaddr_in fromAddr;
    socklen_t addrLen = sizeof(fromAddr);
    
    size_t totalBytesReceived = 0;
    struct FT_Sensor ft;

    while(rclcpp::ok()){
        size_t totalBytesReceived = 0;

        while (totalBytesReceived < BYTE_SIZE) {
            ssize_t bytesReceived = recvfrom(sockfd, buffer + totalBytesReceived, 
                                            BYTE_SIZE - totalBytesReceived, 0, 
                                            (struct sockaddr*)&fromAddr, &addrLen);
            if (bytesReceived < 0) {
                std::cerr << "Error receiving data" << std::endl;
                break;
            }
            totalBytesReceived += bytesReceived;
        }

        if (totalBytesReceived == BYTE_SIZE) {
            sensor_msgs::msg::Imu imu;
            double temp;
            buffer_to_wrench_and_acc(buffer,&wrench.wrench,&imu.linear_acceleration,&imu.angular_velocity,&temp);
            wrench.header.stamp = node.rcl_node()->get_clock()->now();

            wrench.header.frame_id="ft_sensor";
            imu.header=wrench.header;
                      
        
            /*for (ssize_t i = 0; i < totalBytesReceived; ++i) {
                std::cout << "0x" << std::hex << (0xFF & buffer[i]) << " ";
            }*/
    
            /*
            buffer_to_struct(buffer,&ft);
            
            geometry_msgs::msg::WrenchStamped wrench;
            wrench.header.stamp = node.rcl_node()->get_clock()->now();
            wrench.header.frame_id="ft_sensor";
            wrench.wrench.force.x=ft.F[0];
            wrench.wrench.force.y=ft.F[1];
            wrench.wrench.force.z=ft.F[2];
            wrench.wrench.torque.x=ft.T[0];
            wrench.wrench.torque.y=ft.T[1];
            wrench.wrench.torque.z=ft.T[2];
            */
            sensor_pub.publish(wrench);
            imu_pub.publish(imu);


            //printf("\n%f\t%f\t%f\n",wrench.wrench.force.x,wrench.wrench.force.y,wrench.wrench.force.z); 
            lxros::spin_for(std::chrono::milliseconds(10));

            wrench.wrench=filterWrench(wrench.wrench);
            sensor_filt_pub.publish(wrench);

  
        } else {
            std::cerr << "Error: Didn't receive exactly 52 bytes" << std::endl;
        }
    }
    // Step 7: Clean up
    close(sockfd);
    return 0;
}