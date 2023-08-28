// Build with bazel-6.1.0 build --define framework_shared_object=false tensorflow/core/profiler/convert:xplane_to_tools
#include <fstream>
#include <iostream>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/profiler/convert/preprocess_single_host_xplane.h"
#include "tensorflow/core/profiler/convert/process_megascale_dcn.h"
#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_to_json.h"
#include "tensorflow/core/profiler/convert/xplane_to_tools_data.h"
#include "tensorflow/core/profiler/convert/xplane_to_trace_container.h"

absl::StatusOr<std::string> ConvertMultiXSpacesToTraceJson(
    const tensorflow::profiler::SessionSnapshot& session_snapshot,
    const absl::string_view tool_name,
    const tensorflow::profiler::ToolOptions& options) {
  if (session_snapshot.XSpaceSize() != 1) {
    return tensorflow::errors::InvalidArgument(
        "Trace events tool expects only 1 XSpace path but gets ",
        session_snapshot.XSpaceSize());
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<tensorflow::profiler::XSpace> xspace,
                      session_snapshot.GetXSpace(0));
  tensorflow::profiler::PreprocessSingleHostXSpace(xspace.get(),
                                                   /*step_grouping=*/true,
                                                   /*derived_timeline=*/true);
  tensorflow::profiler::ProcessMegascaleDcn(xspace.get());
  tensorflow::profiler::TraceEventsContainer trace_container;
  tensorflow::profiler::ConvertXSpaceToTraceEventsContainer("rank-0", *xspace,
                                                            &trace_container);

  std::string content;
  tensorflow::profiler::JsonTraceOptions jsonOptions;
  tensorflow::profiler::IOBufferAdapter adapter(&content);
  tensorflow::profiler::TraceEventsToJson<
      tensorflow::profiler::IOBufferAdapter,
      tensorflow::profiler::TraceEventsContainer,
      tensorflow::profiler::RawData>(jsonOptions, trace_container, &adapter);
  return content;
}

int main(int argc, char* argv[]) {
  if (argc <= 3) {
    LOG(ERROR) << "USAGE: xplane_to_tools [tool_name] [output_path] [paths ...]"
               << std::endl;
    return 1;
  }
  std::vector<std::string> allArgs(argv, argv + argc);
  std::string tool_name = allArgs[1];
  std::string output_file = allArgs[2];
  allArgs.erase(allArgs.begin());  // pop binary name
  allArgs.erase(allArgs.begin());  // pop tool_name
  allArgs.erase(allArgs.begin());  // pop output_name

  auto status_or_session_snapshot =
      tensorflow::profiler::SessionSnapshot::Create(allArgs,
                                                    /*xspaces=*/std::nullopt);
  if (!status_or_session_snapshot.ok()) {
    LOG(ERROR) << status_or_session_snapshot.status().message();
    return 1;
  }

  tensorflow::profiler::ToolOptions tool_options;
  ::tensorflow::StatusOr<std::string> status_or_tool_data;
  if (tool_name == "trace_json") {
    status_or_tool_data = ConvertMultiXSpacesToTraceJson(
        status_or_session_snapshot.value(), tool_name, tool_options);
  } else {
    status_or_tool_data = tensorflow::profiler::ConvertMultiXSpacesToToolData(
        status_or_session_snapshot.value(), tool_name, tool_options);
  }

  if (!status_or_tool_data.ok()) {
    LOG(ERROR) << status_or_tool_data.status().message();
    return 1;
  }

  std::string output_data = status_or_tool_data.value();

  LOG(INFO) << "Writing output to " + output_file;

  std::ofstream out(output_file);
  out << output_data;
  out.close();
  return 0;
}
