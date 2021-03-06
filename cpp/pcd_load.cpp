#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int
main ()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../../data_pcd/ism_train_wolf.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  // std::cout << "Loaded "
  //           << cloud->width * cloud->height
  //           << " data points from test_pcd.pcd with the following fields: "
  //           << std::endl;
  for (const auto& point: *cloud)
  if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z)) {
    std::cout << "    " << point.x
              << " "    << point.y
              << " "    << point.z << std::endl;
  }
  return (0);
}