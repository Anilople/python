void mouse_callback(GLFWwindow * window, double xpos, double ypos)
{
	static GLboolean firstMouse = true;
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	//GLfloat xpos, ypos;
	GLfloat xoffset = xpos - lastX,
			yoffset = ypos - lastY;
	GLfloat sensitivity = 0.01f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;
	lastX = xpos;
	lastY = ypos;

	pitch += xoffset;
	yaw += yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 direction;
	direction.x = cos(glm::radians(pitch))*cos(glm::radians(yaw));
	direction.y = sin(glm::radians(pitch));
	direction.z = cos(glm::radians(pitch))*sin(glm::radians(yaw));
	cameraFront = glm::normalize(direction);
}