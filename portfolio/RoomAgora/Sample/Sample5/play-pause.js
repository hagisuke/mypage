"use strict";

window.onload = (e)=>{
    const vdo = document.getElementById("main-video");
    const btn = document.getElementById("VideoButton");

    btn.addEventListener("click", (e)=>{
        if(vdo.paused){
            vdo.play();
            e.target.setAttribute("src", "#play-icon");
        }else{
            vdo.pause();
            e.target.setAttribute("src", "#pause-icon");
        }
    });
}