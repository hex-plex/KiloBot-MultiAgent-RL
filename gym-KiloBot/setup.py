import setuptools

setuptools.setup(
    name='gym_kiloBot',
    version='0.0.1',
    description='An OpenAI Gym env for a swarm of kilo bots which have a limited perception of the world',
    packages=setuptools.find_packages(include="gym_kiloBot"),
    intall_required=['gym','pygame','opencv-python'],
    package_data={'':[]}

)
