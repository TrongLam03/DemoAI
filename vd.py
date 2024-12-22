from graphviz import Digraph

# Initialize Digraph object
dot = Digraph(comment='Use Case Diagram - Document and Task Management System')
dot.attr(rankdir='TB', size='8,5')

# Add actors
dot.node('Admin', 'Admin', shape='actor')
dot.node('User', 'User', shape='actor')
 
# Add use cases
dot.node('UC_Login', 'Đăng nhập hệ thống', shape='ellipse')
dot.node('UC_IncomingDocs', 'Quản lý văn bản đến', shape='ellipse')
dot.node('UC_OutgoingDocs', 'Quản lý văn bản đi', shape='ellipse')
dot.node('UC_JobFiles', 'Quản lý hồ sơ công việc', shape='ellipse')
dot.node('UC_SharedFiles', 'Quản lý tài liệu chia sẻ', shape='ellipse')
dot.node('UC_ActivityLog', 'Quản lý nhật ký hoạt động', shape='ellipse')

# Connect Admin to use cases
dot.edge('Admin', 'UC_Login')
dot.edge('Admin', 'UC_IncomingDocs')
dot.edge('Admin', 'UC_OutgoingDocs')
dot.edge('Admin', 'UC_JobFiles')
dot.edge('Admin', 'UC_SharedFiles')
dot.edge('Admin', 'UC_ActivityLog')

# Connect User to use cases
dot.edge('User', 'UC_Login')
dot.edge('User', 'UC_IncomingDocs')
dot.edge('User', 'UC_OutgoingDocs')
dot.edge('User', 'UC_JobFiles')
dot.edge('User', 'UC_SharedFiles')

# Save and render the diagram
output_path = '/mnt/data/Use_Case_Diagram'
dot.render(output_path, format='png', cleanup=True)

output_path + '.png'
