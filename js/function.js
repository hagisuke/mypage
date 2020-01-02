$(function(){
    //モバイルハンバーガーメニュー
    $('.menu-trigger').on('click', function() {
        $(this).toggleClass('active');
        
        if ($('#header_menu').is(':hidden')) $('#header_menu').slideDown();
        else $('#header_menu').slideUp();
        
        return false;
    });
    
    //スクロールしたらヘッダー背景
    //var wH = $(window).height();
    var wH = $('#header-point').offset().top;
    $(window).on('load scroll', function() {
        var value = $(this).scrollTop();
        if ( value > wH ) {
            $('header').addClass('header_background');
        } else {
            $('header').removeClass('header_background');
        }
    });
});
