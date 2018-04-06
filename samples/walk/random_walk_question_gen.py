# Copyright (c) 2017, IGLU consortium
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import os
import sys
import logging
import numpy as np
import collections
import matplotlib.pyplot as plt
import time

from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import LVector3f, Vec3
from panda3d.ai import *

from home_platform.constants import MODEL_CATEGORY_COLOR_MAPPING
from home_platform.env import BasicEnvironment
from home_platform.rendering import Panda3dSemanticsRenderer
from home_platform.semantic import SuncgSemantics
from home_platform.utils import Viewer
from question_gen.utils import create_metadata_from_frames

TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data", "suncg")

logger = logging.getLogger(__name__)


def main():
    env = BasicEnvironment(houseId="0004d52d1aeeb8ae6de39d6bd993e992", suncgDatasetRoot=TEST_SUNCG_DATA_DIR,
                           realtime=True)

    # a, b = env.generateOccupancyMap()
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show(block=False)

    env.setAgentOrientation((60.0, 0.0, 0.0))
    env.setAgentPosition((42, -39, 1.2))
    env.renderWorld.showRoomLayout(showCeilings=False, showWalls=True, showFloors=True)

    viewer = Viewer(env.scene, interactive=False, showPosition=True, fov=50.)

    # Find agent and reparent camera to it
    agent = env.agentRbNp
    viewer.camera.reparentTo(agent)
    viewer.capture_video()

    # ai_world = AIWorld(env.scene.scene)
    # ai_char = AICharacter("seeker", agent, 60, 0.5, 5)
    # ai_world.addAiChar(ai_char)
    # ai_behaviors = ai_char.getAiBehaviors()
    # ai_behaviors.seek(Vec3(35, -39, 1.0))
    #
    # def update_ai_world(task):
    #     ai_world.update()
    #     return task.cont

    # taskMgr.add(update_ai_world, "AIUpdate")
    #
    # while True:
    #
    #     viewer.step()
    #     env.step()

    renderer = Panda3dSemanticsRenderer(env.scene, TEST_SUNCG_DATA_DIR, mode='offscreen', segment_by_instance=True)
    renderer.cameras[0].reparentTo(agent)
    renderer.showRoomLayout(showCeilings=False)
    images = []

    linearVelocity = np.zeros(3)
    angularVelocity = np.zeros(3)
    rotationStepCounter = -1
    rotationsStepDuration = 40
    try:
        while True:

            # Constant speed forward (Y-axis)
            linearVelocity = LVector3f(0.0, 1.0, 0.0)
            env.setAgentLinearVelocity(linearVelocity)

            # Randomly change angular velocity (rotation around Z-axis)
            if rotationStepCounter > rotationsStepDuration:
                # End of rotation
                rotationStepCounter = -1
                angularVelocity = np.zeros(3)
            elif rotationStepCounter >= 0:
                # During rotation
                rotationStepCounter += 1
            else:
                # No rotation, initiate at random
                if np.random.random() > 0.5:
                    angularVelocity = np.zeros(3)
                    angularVelocity[2] = np.random.uniform(low=-np.pi, high=np.pi)
                    rotationStepCounter = 0
            env.setAgentAngularVelocity(angularVelocity)

            # Simulate
            env.step()
            viewer.step()

            image = renderer.getRgbaImages()['agent-0']
            images.append(image)

            if viewer.stop:
                postprocess_images(images, env.scene, renderer)
                sys.exit(0)

    except KeyboardInterrupt:
        postprocess_images(images, env.scene, renderer)
        pass

    viewer.destroy()

    return 0


def postprocess_images(images, scene, segmentation_renderer):
    create_metadata_from_frames(images, scene, segmentation_renderer)


if __name__ == "__main__":
    sys.exit(main())
