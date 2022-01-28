Attribute VB_Name = "Example"
Sub usage()
    ' �u�}�`��I������ɂ́A�\�����A�N�e�B�u�łȂ���΂����܂���v�G���[���o���ꍇ�́A
    ' �X���C�h���N���b�N���Ă���}�N�������s���Ă݂�

    ' �I�u�W�F�N�g�^�̕ϐ���錾
    Dim dm As DrawModel
    
    ' �C���X�^���X����
    Set dm = New DrawModel

    ' �J�n�u���b�N
    Call dm.StartBlock(name:="start", start_w:=10, start_h:=100)
    
    ' �e�L�X�g�u���b�N
    Call dm.TextBlock(name:="text", from_name:="start", offset_x:=5, offset_y:=0, _
        txt_w:=80, txt_h:=30, txt_content:="<bos>", font_size:=18)

    ' �摜�u���b�N�i�t�@�C����������Ȃ��G���[���o���ꍇ�͐�΃p�X�Ŏw�肵�Ă݂�j
    Call dm.ImageBlock(name:="image", from_name:="text", offset_x:=5, offset_y:=0, _
        img_w:=80, img_h:=80, img_path:="figA.jpg", img_aspect:=msoTrue)
    
    ' �L���[�u�u���b�N
    Call dm.CubeBlock(name:="cube1", from_name:="image", offset_x:=5, offset_y:=0, _
        cube_w:=60, cube_h:=60, cube_d:=0, txt_content:="3D", font_size:=18)
    Call dm.CubeBlock(name:="cube2", from_name:="cube1", offset_x:=5, offset_y:=0, _
        cube_w:=20, cube_h:=100, cube_d:=100, txt_content:="", font_size:=18, depth_mode:=msoTrue)
        
    ' �����u���b�N
    Call dm.ProcessBlock(name:="process", from_name:="cube2", offset_x:=5, offset_y:=0, _
        process_w:=80, process_h:=40, txt_content:="module", font_size:=18, txt_orientation:="h")
        
    ' �_�E���T���v���u���b�N
    Call dm.DownsampleBlock(name:="enc", from_name:="process", offset_x:=5, offset_y:=0, _
        process_w:=80, process_h:=100, txt_content:="enc", font_size:=18, orientation:="h")
        
    ' �A�b�v�T���v���u���b�N
    Call dm.UpsampleBlock(name:="dec", from_name:="enc", offset_x:=0, offset_y:=0, _
        process_w:=80, process_h:=100, txt_content:="dec", font_size:=18, orientation:="h")
        
    ' �e�L�X�g�����w�u���b�N
    Call dm.LayerBlock(name:="layer1", from_name:="dec", offset_x:=50, offset_y:=0, _
        layer_w:=10, layer_h:=200)
        
    ' �e�L�X�g�t���w�u���b�N
    Call dm.LayerAndTextBlock(name:="layer2", from_name:="layer1", offset_x:=30, offset_y:=0, _
        layer_w:=10, layer_h:=200, left_txt_w:=100, left_txt_h:=30, left_txt_content:="128x128", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="64", txt_offset:=2, font_size:=18)
    
    ' �}�[�N�u���b�N
    Call dm.MarkBlock(name:="mark", from_name:="layer2", offset_x:=200, offset_y:=0, _
        mark_w:=20, mark_h:=20, mark_content:="+", line_weight:=2, font_size:=32)
    
    ' ���u���b�N
    Call dm.ArrowBlock(name:="right1", from_name:="mark", offset_x:=5, offset_y:=0, _
        arrow_w:=40, arrow_h:=20, orientation:="Right")
    Call dm.ArrowBlock(name:="left1", from_name:="mark", offset_x:=-5, offset_y:=0, _
        arrow_w:=40, arrow_h:=20, orientation:="Left")
    Call dm.ArrowBlock(name:="up1", from_name:="mark", offset_x:=0, offset_y:=-50, _
        arrow_w:=20, arrow_h:=40, orientation:="Up")
    Call dm.ArrowBlock(name:="down1", from_name:="mark", offset_x:=0, offset_y:=50, _
        arrow_w:=20, arrow_h:=40, orientation:="Down")
    
    ' ���u���b�N�R�l�N�^
    Call dm.ArrowBlockConnector(name:="right2", from_name:="layer1", to_name:="layer2", _
        offset_x:=5, offset_y:=0, arrow_weight:=20, orientation:="Right")
    Call dm.ArrowBlockConnector(name:="left2", from_name:="layer2", to_name:="layer1", _
        offset_x:=-5, offset_y:=0, arrow_weight:=20, orientation:="Left")
    Call dm.ArrowBlockConnector(name:="up2", from_name:="mark", to_name:="up1", _
        offset_x:=0, offset_y:=-5, arrow_weight:=20, orientation:="Up")
    Call dm.ArrowBlockConnector(name:="down2", from_name:="up1", to_name:="mark", _
        offset_x:=0, offset_y:=5, arrow_weight:=20, orientation:="Down")
    
    ' �����R�l�N�^
    Call dm.ArrowLineConnector(name:="right3", from_name:="layer2", to_name:="mark", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="left3", from_name:="mark", to_name:="layer2", line_weight:=2, orientation:="Left")
    Call dm.ArrowLineConnector(name:="up3", from_name:="down1", to_name:="mark", line_weight:=2, orientation:="Up")
    Call dm.ArrowLineConnector(name:="down3", from_name:="mark", to_name:="down1", line_weight:=2, orientation:="Down")
    
    ' ��������R�l�N�^
    Call dm.BranchArrowLineConnector(name:="branch", from_name:="right3", to_name:="down1", line_weight:=2)
    
    ' ���������R�l�N�^
    Call dm.MergeArrowLineConnector(name:="merge", from_name:="down1", to_name:="right1", line_weight:=2)
    
    ' �X�L�b�v�����R�l�N�^
    Call dm.SkipArrowLineConnector(name:="skip", from_name:="right3", to_name:="right1", line_h:=120, line_weight:=2)
    
    ' �J�M�����R�l�N�^
    Call dm.ElbowArrowLineConnector(name:="elbow", from_name:="layer1", to_name:="layer2", line_weight:=2)

    ' �I�u�W�F�N�g�j��
    Set dm = Nothing
End Sub

Sub simple_resnet()
    Dim dm As DrawModel
    Set dm = New DrawModel

    ' �܂��u���b�N����ׂ�
    Call dm.StartBlock(name:="start", start_w:=30, start_h:=100)
    Call dm.TextBlock(name:="text1", from_name:="start", offset_x:=30, offset_y:=0, _
        txt_w:=80, txt_h:=30, txt_content:="Input", font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer1", from_name:="text1", offset_x:=30, offset_y:=0, _
        layer_w:=10, layer_h:=200, left_txt_w:=80, left_txt_h:=30, left_txt_content:="128x128", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="64", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer2", from_name:="layer1", offset_x:=30, offset_y:=0, _
        layer_w:=10, layer_h:=160, left_txt_w:=60, left_txt_h:=30, left_txt_content:="64x64", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="128", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer3", from_name:="layer2", offset_x:=60, offset_y:=0, _
        layer_w:=15, layer_h:=120, left_txt_w:=60, left_txt_h:=30, left_txt_content:="32x32", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="256", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer3_2", from_name:="layer2", offset_x:=60, offset_y:=150, _
        layer_w:=15, layer_h:=120, left_txt_w:=60, left_txt_h:=30, left_txt_content:="32x32", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="256", txt_offset:=1, font_size:=18)
    Call dm.MarkBlock(name:="mark1", from_name:="layer3", offset_x:=30, offset_y:=0, _
        mark_w:=20, mark_h:=20, mark_content:="+", line_weight:=2, font_size:=32)
    Call dm.LayerAndTextBlock(name:="layer4", from_name:="mark1", offset_x:=60, offset_y:=0, _
        layer_w:=20, layer_h:=110, left_txt_w:=60, left_txt_h:=30, left_txt_content:="16x16", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="512", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer5", from_name:="layer4", offset_x:=30, offset_y:=0, _
        layer_w:=20, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="8x8", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="512", txt_offset:=1, font_size:=18)
    Call dm.MarkBlock(name:="mark2", from_name:="layer5", offset_x:=30, offset_y:=0, _
        mark_w:=20, mark_h:=20, mark_content:="+", line_weight:=2, font_size:=32)
    Call dm.LayerAndTextBlock(name:="layer6", from_name:="mark2", offset_x:=30, offset_y:=0, _
        layer_w:=20, layer_h:=90, left_txt_w:=50, left_txt_h:=30, left_txt_content:="4x4", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="512", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer7", from_name:="layer6", offset_x:=30, offset_y:=0, _
        layer_w:=40, layer_h:=10, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="8192", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer8", from_name:="layer7", offset_x:=30, offset_y:=0, _
        layer_w:=20, layer_h:=10, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="10", txt_offset:=1, font_size:=18)
    Call dm.TextBlock(name:="text2", from_name:="layer8", offset_x:=30, offset_y:=0, _
        txt_w:=80, txt_h:=30, txt_content:="Output", font_size:=18)
    
    ' ���ɃR�l�N�^�Ōq��
    Call dm.ArrowLineConnector(name:="right1", from_name:="text1", to_name:="layer1", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right2", from_name:="layer1", to_name:="layer2", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right3", from_name:="layer2", to_name:="layer3", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right4", from_name:="layer3", to_name:="mark1", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right5", from_name:="mark1", to_name:="layer4", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right6", from_name:="layer4", to_name:="layer5", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right7", from_name:="layer5", to_name:="mark2", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right8", from_name:="mark2", to_name:="layer6", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right9", from_name:="layer6", to_name:="layer7", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right10", from_name:="layer7", to_name:="layer8", line_weight:=2, orientation:="Right")
    Call dm.ArrowLineConnector(name:="right11", from_name:="layer8", to_name:="text2", line_weight:=2, orientation:="Right")
    Call dm.BranchArrowLineConnector(name:="branch", from_name:="right3", to_name:="layer3_2", line_weight:=2)
    Call dm.MergeArrowLineConnector(name:="merge", from_name:="layer3_2", to_name:="mark1", line_weight:=2)
    Call dm.SkipArrowLineConnector(name:="skip", from_name:="right5", to_name:="mark2", line_h:=90, line_weight:=2)
    
    Set dm = Nothing
End Sub

Sub simple_unet()
    Dim dm As DrawModel
    Set dm = New DrawModel

    ' �܂��u���b�N����ׂ�
    Call dm.StartBlock(name:="start", start_w:=50, start_h:=100)
    Call dm.TextBlock(name:="text1", from_name:="start", offset_x:=0, offset_y:=-150, _
        txt_w:=80, txt_h:=30, txt_content:="Input", font_size:=18)
    
    Call dm.LayerAndTextBlock(name:="layer1_1", from_name:="text1", offset_x:=0, offset_y:=0, _
        layer_w:=5, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="1", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right1_1", from_name:="layer1_1", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer1_2", from_name:="right1_1", offset_x:=5, offset_y:=0, _
        layer_w:=10, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="64", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right1_2", from_name:="layer1_2", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer1_3", from_name:="right1_2", offset_x:=5, offset_y:=0, _
        layer_w:=10, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="64", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="down1", from_name:="layer1_3", offset_x:=0, offset_y:=5, _
        arrow_w:=20, arrow_h:=20, orientation:="Down")
        
    Call dm.LayerAndTextBlock(name:="layer2_1", from_name:="down1", offset_x:=-15, offset_y:=95, _
        layer_w:=10, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="64", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right2_1", from_name:="layer2_1", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer2_2", from_name:="right2_1", offset_x:=5, offset_y:=0, _
        layer_w:=20, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="128", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right2_2", from_name:="layer2_2", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer2_3", from_name:="right2_2", offset_x:=5, offset_y:=0, _
        layer_w:=20, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="128", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="down2", from_name:="layer2_3", offset_x:=0, offset_y:=5, _
        arrow_w:=20, arrow_h:=20, orientation:="Down")
        
    Call dm.LayerAndTextBlock(name:="layer3_1", from_name:="down2", offset_x:=-20, offset_y:=70, _
        layer_w:=20, layer_h:=50, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="128", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right3_1", from_name:="layer3_1", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer3_2", from_name:="right3_1", offset_x:=5, offset_y:=0, _
        layer_w:=40, layer_h:=50, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="256", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right3_2", from_name:="layer3_2", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer3_3", from_name:="right3_2", offset_x:=5, offset_y:=0, _
        layer_w:=40, layer_h:=50, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="256", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="up1", from_name:="layer3_3", offset_x:=0, offset_y:=-35, _
        arrow_w:=20, arrow_h:=20, orientation:="Up")
        
    Call dm.LayerAndTextBlock(name:="layer4_1", from_name:="up1", offset_x:=-20, offset_y:=-65, _
        layer_w:=20, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer4_1_concat", from_name:="layer4_1", offset_x:=-41, offset_y:=0, _
        layer_w:=20, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="256", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right4_1", from_name:="layer4_1", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer4_2", from_name:="right4_1", offset_x:=5, offset_y:=0, _
        layer_w:=20, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="128", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right4_2", from_name:="layer4_2", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer4_3", from_name:="right4_2", offset_x:=5, offset_y:=0, _
        layer_w:=20, layer_h:=100, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="128", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="up2", from_name:="layer4_3", offset_x:=0, offset_y:=-35, _
        arrow_w:=20, arrow_h:=20, orientation:="Up")
        
    Call dm.LayerAndTextBlock(name:="layer5_1", from_name:="up2", offset_x:=-15, offset_y:=-90, _
        layer_w:=10, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="", txt_offset:=1, font_size:=18)
    Call dm.LayerAndTextBlock(name:="layer5_1_concat", from_name:="layer5_1", offset_x:=-21, offset_y:=0, _
        layer_w:=10, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="128", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right5_1", from_name:="layer5_1", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer5_2", from_name:="right5_1", offset_x:=5, offset_y:=0, _
        layer_w:=10, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="64", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right5_2", from_name:="layer5_2", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer5_3", from_name:="right5_2", offset_x:=5, offset_y:=0, _
        layer_w:=10, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="64", txt_offset:=1, font_size:=18)
    Call dm.ArrowBlock(name:="right5_3", from_name:="layer5_3", offset_x:=5, offset_y:=0, _
        arrow_w:=20, arrow_h:=20, orientation:="Right")
    Call dm.LayerAndTextBlock(name:="layer5_4", from_name:="right5_3", offset_x:=5, offset_y:=0, _
        layer_w:=5, layer_h:=150, left_txt_w:=50, left_txt_h:=30, left_txt_content:="", _
        top_txt_w:=60, top_txt_h:=30, top_txt_content:="2", txt_offset:=1, font_size:=18)
    Call dm.TextBlock(name:="text2", from_name:="layer5_4", offset_x:=0, offset_y:=0, _
        txt_w:=80, txt_h:=30, txt_content:="Output", font_size:=18)

    ' ���ɃR�l�N�^�Ōq��
    Call dm.ArrowBlockConnector(name:="skip1", from_name:="layer1_3", to_name:="layer5_1_concat", _
        offset_x:=5, offset_y:=0, arrow_weight:=20, orientation:="Right")
    Call dm.ArrowBlockConnector(name:="skip2", from_name:="layer2_3", to_name:="layer4_1_concat", _
        offset_x:=5, offset_y:=0, arrow_weight:=20, orientation:="Right")

    Set dm = Nothing
End Sub

