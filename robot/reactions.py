import asyncio
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class Reactions:
    """Head and antenna movement reactions for focus notifications"""
    
    def __init__(self, motors):
        self.motors = motors
    
    async def look_at(self, pitch: float = 0, roll: float = 0, yaw: float = 0, duration: float = 1.0):
        """Look in a specific direction"""
        self.motors.goto_head(pitch=pitch, roll=roll, yaw=yaw, duration=duration)
        await asyncio.sleep(duration)
    
    async def shake_head(self, times: int = 2, duration: float = 0.3):
        """Disapproval head shake - use smaller angles"""
        for _ in range(times):
            self.motors.goto_head(yaw=15, duration=duration/2)
            await asyncio.sleep(duration/2)
            self.motors.goto_head(yaw=-15, duration=duration/2)
            await asyncio.sleep(duration/2)
        self.motors.goto_head(yaw=0, duration=duration/2)
        await asyncio.sleep(duration/2)
    
    async def nod(self, times: int = 2, duration: float = 0.4):
        """Approval nod - use smaller angles"""
        for _ in range(times):
            self.motors.goto_head(pitch=10, duration=duration/2)
            await asyncio.sleep(duration/2)
            self.motors.goto_head(pitch=-5, duration=duration/2)
            await asyncio.sleep(duration/2)
        self.motors.goto_head(pitch=0, duration=duration/2)
        await asyncio.sleep(duration/2)
    
    async def tilt_head(self, angle: float = 15, duration: float = 0.5):
        """Questioning head tilt - use smaller angle"""
        self.motors.goto_head(roll=angle, duration=duration)
        await asyncio.sleep(duration)
        self.motors.goto_head(roll=0, duration=duration)
        await asyncio.sleep(duration)
    
    async def look_away(self, duration: float = 2.0):
        """Look away (user is distracted)"""
        self.motors.goto_head(yaw=90, pitch=-10, duration=1.0)
        await asyncio.sleep(duration)
        self.motors.goto_head(yaw=0, pitch=0, duration=1.0)
        await asyncio.sleep(1.0)
    
    async def wave_antenna_left_right(self, times: int = 3, speed: float = 0.2):
        """Wave antenna left-right: 'Focus!' warning"""
        for _ in range(times):
            self.motors.move_antenna("left", 30, duration=speed)
            await asyncio.sleep(speed)
            self.motors.move_antenna("left", -30, duration=speed)
            await asyncio.sleep(speed)
        self.motors.move_antenna("left", 0, duration=speed)
        await asyncio.sleep(speed)
    
    async def wave_antenna_up(self, times: int = 2, speed: float = 0.3):
        """Wave antenna up: greeting"""
        for _ in range(times):
            self.motors.move_antenna("right", 45, duration=speed)
            await asyncio.sleep(speed)
            self.motors.move_antenna("right", 0, duration=speed)
            await asyncio.sleep(speed)
    
    async def antenna_uh_oh(self, times: int = 1, speed: float = 0.2):
        """Uh oh - antenna waves from 0 to 180 and back"""
        for _ in range(times):
            # Left antenna goes up (0 -> 180 -> 0)
            self.motors.move_antenna("left", 90, duration=speed)
            await asyncio.sleep(speed)
            self.motors.move_antenna("left", 0, duration=speed)
            await asyncio.sleep(speed)
    
    async def antenna_boom(self, speed: float = 0.15):
        """Both antennas wave: strong warning"""
        for _ in range(2):
            self.motors.move_antenna("left", 45, duration=speed)
            self.motors.move_antenna("right", 45, duration=speed)
            await asyncio.sleep(speed)
            self.motors.move_antenna("left", -45, duration=speed)
            self.motors.move_antenna("right", -45, duration=speed)
            await asyncio.sleep(speed)
        self.motors.move_antenna("left", 0, duration=speed)
        self.motors.move_antenna("right", 0, duration=speed)
        await asyncio.sleep(speed)
    
    async def greeting(self):
        """Full greeting sequence - simplified for simulation"""
        await self.wave_antenna_up(times=2)
    
    async def uh_oh(self):
        """Uh oh reaction - antenna wave"""
        await self.antenna_uh_oh(times=1)
    
    async def warning(self):
        """Warning reaction for distraction"""
        await self.wave_antenna_left_right(times=3)
        await self.tilt_head(angle=25)
    
    async def scold(self):
        """Strong warning for serious distraction"""
        await self.antenna_boom()
        await self.shake_head(times=2)
        await self.look_away(duration=3.0)
    
    async def focused(self):
        """Reaction when user is focused"""
        await self.nod(times=1)
    
    async def execute_reaction(self, reaction: str):
        """Execute a named reaction"""
        logger.info(f"Executing reaction: {reaction}")
        
        reaction_map = {
            "none": lambda: asyncio.sleep(0),
            "greeting": self.greeting,
            "warning": self.warning,
            "scold": self.scold,
            "focused": self.focused,
            "look_away": self.look_away,
            "shake_head": lambda: self.shake_head(),
            "nod": lambda: self.nod(),
            "tilt": lambda: self.tilt_head(),
            "uh_oh": self.uh_oh,
        }
        
        reaction_func = reaction_map.get(reaction)
        if reaction_func:
            await reaction_func()
        else:
            logger.warning(f"Unknown reaction: {reaction}")


class ReactionsWithAudio(Reactions):
    """Reactions with audio playback"""
    
    def __init__(self, motors, speaker):
        super().__init__(motors)
        self.speaker = speaker
    
    async def play_and_react(self, text: str, reaction: str):
        """Play TTS audio and execute reaction"""
        if text:
            self.speaker.play_tts(text)
        
        await self.execute_reaction(reaction)
    
    async def warning_with_voice(self, text: str = "Hey, you should focus on your task!"):
        """Warning with voice"""
        await self.play_and_react(text, "warning")
    
    async def scold_with_voice(self, text: str = "Put that away and focus!"):
        """Strong warning with voice"""
        await self.play_and_react(text, "scold")
    
    async def greeting_with_voice(self, text: str = "Hi! Let's get some work done!"):
        """Greeting with voice"""
        await self.play_and_react(text, "greeting")
