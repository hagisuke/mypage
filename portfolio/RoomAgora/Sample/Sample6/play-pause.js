"use strict";

window.onload = (e)=>{
    const snd = document.getElementById("MainSound");
    const btn = document.getElementById("AudioButton");

    btn.addEventListener("click", (e)=>{
        const sound = snd.components.sound;
        if(sound.isPlaying){
            sound.pauseSound();
            e.target.setAttribute("src", "#pause-icon");
        }else{
            sound.playSound();
            e.target.setAttribute("src", "#play-icon");
        }
    });
}