$(function(){
    //モバイルハンバーガーメニュー
    $('.menu-trigger').on('click', function() {
        $(this).toggleClass('active');
        
        if ($('#header_menu').is(':hidden')) $('#header_menu').slideDown();
        else $('#header_menu').slideUp();
        
        return false;
    });
});
