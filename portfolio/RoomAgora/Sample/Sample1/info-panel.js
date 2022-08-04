/* global AFRAME */
AFRAME.registerComponent('info-panel', {
  init: function () {
    var buttonEl = document.querySelector('.menu-button');
    this.onMenuButtonClick = this.onMenuButtonClick.bind(this);
    buttonEl.addEventListener('click', this.onMenuButtonClick);
  },

  onMenuButtonClick: function (evt) {
    if(this.el.object3D.visible == false) {
        this.el.object3D.scale.set(1, 1, 1);
        this.el.object3D.visible = true;
    } else {
        this.el.object3D.scale.set(0.001, 0.001, 0.001);
        this.el.object3D.visible = false;
    }
  }
});