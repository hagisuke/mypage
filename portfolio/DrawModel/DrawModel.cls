VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "DrawModel"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Option Explicit

' �����o�ϐ�
Public prs As PowerPoint.Presentation
Public sld As PowerPoint.Slide

' ����������
Private Sub Class_Initialize()
    ' �J���Ă���v���[���e�[�V������prs�Ɋi�[
    Set Me.prs = ActivePresentation
    ' �����X���C�h�ǉ�
    Set Me.sld = Me.prs.Slides.Add(Me.prs.Slides.Count + 1, ppLayoutBlank)
    Me.sld.Select
    Debug.Print "initalized"
End Sub

' �I��������
Private Sub Class_Terminate()
    ActiveWindow.Selection.Unselect
    Debug.Print "terminated"
End Sub

' �J�n�u���b�N
Public Sub StartBlock(name As String, start_w As Single, start_h As Single)
    ' �X���C�h�̍��[�E�^�񒆂̍����ɓ����ȃu���b�N��u��
    '
    ' Args
    ' name: �J�n�u���b�N�̖��O
    ' start_w: �J�n�u���b�N�̕�
    ' start_h: �J�n�u���b�N�̍���
    Dim start As PowerPoint.Shape
    Set start = Me.sld.Shapes.AddShape( _
        Type:=msoShapeRectangle, _
        Left:=0, _
        Top:=(Me.sld.Master.Height - start_h) / 2, _
        Width:=start_w, _
        Height:=start_h)
    With start
        .Select
        .name = name
        .Fill.Visible = msoFalse
        .Line.Visible = msoFalse
    End With
End Sub

' �e�L�X�g�u���b�N
Public Sub TextBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    txt_w As Single, txt_h As Single, txt_content As String, font_size As Single)
    ' �����I�u�W�F�N�g�̉E�[����Ƀe�L�X�g�{�b�N�X��u��
    '
    ' Args
    ' name: �e�L�X�g�u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' txt_w: �e�L�X�g�u���b�N�̕�
    ' txt_h: �e�L�X�g�u���b�N�̍���
    ' txt_content: �e�L�X�g�̒��g
    ' font_size: �t�H���g�T�C�Y
    Dim txt As PowerPoint.Shape
    Set txt = Me.sld.Shapes.AddTextbox( _
        orientation:=msoTextOrientationHorizontal, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - txt_h) / 2 + offset_y, _
        Width:=txt_w, _
        Height:=txt_h)
    With txt
        .Select
        .name = name
        .TextFrame.TextRange = txt_content
        .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
        .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
        .TextFrame.AutoSize = ppAutoSizeNone
        .TextEffect.FontSize = font_size
    End With
End Sub

' �摜�u���b�N
Public Sub ImageBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    img_w As Single, img_h As Single, img_path As String, Optional img_aspect As MsoTriState = msoTrue)
    ' �����I�u�W�F�N�g�̉E�[����ɉ摜��u��
    '
    ' Args
    ' name: �摜�u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' img_w: �摜�̕�
    ' img_h: �摜�̍���
    ' img_path: �摜�t�@�C���ւ̃p�X
    ' img_aspect: �c����Œ�̃t���O�iimg_h�ɍ��킹��BDefault: msoTrue�j
    Dim img As PowerPoint.Shape
    Set img = Me.sld.Shapes.AddPicture(FileName:=img_path, _
        LinkToFile:=msoFalse, _
        SaveWithDocument:=msoTrue, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - img_h) / 2 + offset_y)
    With img
        .LockAspectRatio = img_aspect ' �c������Œ�
        .Select
        .name = name
        ' �摜�T�C�Y�ύX
        .Width = img_w
        .Height = img_h
    End With
End Sub

' �L���[�u�u���b�N
Public Sub CubeBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    cube_w As Single, cube_h As Single, cube_d As Single, txt_content As String, font_size As Single, Optional depth_mode As MsoTriState = msoFalse)
    ' �����I�u�W�F�N�g�̉E�[����ɗ����́i�����́j��u��
    '
    ' Args
    ' name: �L���[�u�u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' cube_w: �L���[�u�u���b�N�̕�
    ' cube_h: �L���[�u�u���b�N�̍���
    ' cube_d: �L���[�u�u���b�N�̉��s��
    ' txt_content: �e�L�X�g�̒��g
    ' font_size: �t�H���g�T�C�Y
    ' depth_mode: msoTrue�Ȃ�L���[�u�𕝁E�����E���s���Ŏw�肵�AmsoFalse�Ȃ�I�u�W�F�N�g�̕��E�����Ŏw��iDefault: msoFalse�j
    Dim cube As PowerPoint.Shape
    If depth_mode = msoTrue Then
        Set cube = Me.sld.Shapes.AddShape( _
            Type:=msoShapeCube, _
            Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
            Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - (cube_h + 0.75 * cube_d / Sqr(2))) / 2 + offset_y, _
            Width:=cube_w + 0.75 * cube_d / Sqr(2), _
            Height:=cube_h + 0.75 * cube_d / Sqr(2))
    Else
        Set cube = Me.sld.Shapes.AddShape( _
            Type:=msoShapeCube, _
            Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
            Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - cube_h) / 2 + offset_y, _
            Width:=cube_w, _
            Height:=cube_h)
    End If
    With cube
        .Select
        .name = name
        If depth_mode = msoTrue Then
            .Adjustments.Item(1) = (.Height - cube_h) / .Width
        End If
        .Fill.ForeColor.RGB = RGB(0, 175, 236)
        .Line.Visible = msoFalse
        .TextFrame.TextRange = txt_content
        .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
        .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
        .TextFrame.AutoSize = ppAutoSizeNone
        .TextEffect.FontSize = font_size
    End With
End Sub

' �����u���b�N
Public Sub ProcessBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    process_w As Single, process_h As Single, txt_content As String, font_size As Single, txt_orientation As String)
    ' �����I�u�W�F�N�g�̉E�[����ɏ����u���b�N��u��
    '
    ' Args
    ' name: �����u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' process_w: �����u���b�N�̕�
    ' process_h: �����u���b�N�̍���
    ' txt_content: �e�L�X�g�̒��g
    ' font_size: �t�H���g�T�C�Y
    ' txt_orientation: �e�L�X�g�̕����i"h"��"v"�j
    Dim process As PowerPoint.Shape
    Set process = Me.sld.Shapes.AddShape( _
        Type:=msoShapeRectangle, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - process_h) / 2 + offset_y, _
        Width:=process_w, _
        Height:=process_h)
    If txt_orientation = "h" Then
        With process
            .Select
            .name = name
            .Fill.ForeColor.RGB = RGB(246, 173, 60)
            .Line.Visible = msoFalse
            .TextFrame.TextRange = txt_content
            .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
            .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
            .TextFrame.AutoSize = ppAutoSizeNone
            .TextEffect.FontSize = font_size
        End With
    ElseIf txt_orientation = "v" Then
        With process
            .Select
            .name = name
            .Fill.ForeColor.RGB = RGB(246, 173, 60)
            .Line.Visible = msoFalse
            .TextFrame.TextRange = txt_content
            .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
            .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
            .TextFrame.AutoSize = ppAutoSizeNone
            .TextEffect.FontSize = font_size
            .TextFrame.orientation = msoTextOrientationUpward
        End With
    End If
End Sub

' �_�E���T���v���u���b�N
Public Sub DownsampleBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    process_w As Single, process_h As Single, txt_content As String, font_size As Single, orientation As String)
    ' �����I�u�W�F�N�g�̉E�[����Ƀ_�E���T���v���u���b�N��u��
    '
    ' Args
    ' name: �_�E���T���v���u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' process_w: �_�E���T���v���u���b�N�̕�
    ' process_h: �_�E���T���v���u���b�N�̍���
    ' txt_content: �e�L�X�g�̒��g
    ' font_size: �t�H���g�T�C�Y
    ' orientation: �I�u�W�F�N�g�̕����i"h"��"v"�j
    Dim process As PowerPoint.Shape
    If orientation = "h" Then
        Set process = Me.sld.Shapes.AddShape( _
            Type:=msoShapeTrapezoid, _
            Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x + (process_w - process_h) / 2, _
            Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - process_h) / 2 + offset_y - (process_w - process_h) / 2, _
            Width:=process_h, _
            Height:=process_w)
        With process
            .Select
            .Fill.ForeColor.RGB = RGB(246, 173, 60)
            .Line.Visible = msoFalse
            .Rotation = 90
        End With
    ElseIf orientation = "v" Then
        Set process = Me.sld.Shapes.AddShape( _
            Type:=msoShapeTrapezoid, _
            Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
            Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - process_h) / 2 + offset_y, _
            Width:=process_w, _
            Height:=process_h)
        With process
            .Select
            .Fill.ForeColor.RGB = RGB(246, 173, 60)
            .Line.Visible = msoFalse
            .Flip msoFlipVertical
        End With
    End If
    
    ' ���W�n�ێ��̂��߂̃x�[�X�ǉ�
    Dim base As PowerPoint.Shape
    Set base = Me.sld.Shapes.AddShape( _
        Type:=msoShapeRectangle, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - process_h) / 2 + offset_y, _
        Width:=process_w, _
        Height:=process_h)
    With base
        .Select
        .name = name
        .Fill.Visible = msoFalse
        .Line.Visible = msoFalse
        .TextFrame.TextRange = txt_content
        .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
        .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
        .TextFrame.AutoSize = ppAutoSizeNone
        .TextEffect.FontSize = font_size
    End With
End Sub

' �A�b�v�T���v���u���b�N
Public Sub UpsampleBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    process_w As Single, process_h As Single, txt_content As String, font_size As Single, orientation As String)
    ' �����I�u�W�F�N�g�̉E�[����ɃA�b�v�T���v���u���b�N��u��
    '
    ' Args
    ' name: �A�b�v�T���v���u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' process_w: �A�b�v�T���v���u���b�N�̕�
    ' process_h: �A�b�v�T���v���u���b�N�̍���
    ' txt_content: �e�L�X�g�̒��g
    ' font_size: �t�H���g�T�C�Y
    ' orientation: �I�u�W�F�N�g�̕����i"h"��"v"�j
    Dim process As PowerPoint.Shape
    If orientation = "h" Then
        Set process = Me.sld.Shapes.AddShape( _
            Type:=msoShapeTrapezoid, _
            Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x + (process_w - process_h) / 2, _
            Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - process_h) / 2 + offset_y - (process_w - process_h) / 2, _
            Width:=process_h, _
            Height:=process_w)
        With process
            .Select
            .Fill.ForeColor.RGB = RGB(246, 173, 60)
            .Line.Visible = msoFalse
            .Rotation = -90
        End With
    ElseIf orientation = "v" Then
        Set process = Me.sld.Shapes.AddShape( _
            Type:=msoShapeTrapezoid, _
            Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
            Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - process_h) / 2 + offset_y, _
            Width:=process_w, _
            Height:=process_h)
        With process
            .Select
            .Fill.ForeColor.RGB = RGB(246, 173, 60)
            .Line.Visible = msoFalse
        End With
    End If
    
    ' ���W�n�ێ��̂��߂̃x�[�X�ǉ�
    Dim base As PowerPoint.Shape
    Set base = Me.sld.Shapes.AddShape( _
        Type:=msoShapeRectangle, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - process_h) / 2 + offset_y, _
        Width:=process_w, _
        Height:=process_h)
    With base
        .Select
        .name = name
        .Fill.Visible = msoFalse
        .Line.Visible = msoFalse
        .TextFrame.TextRange = txt_content
        .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
        .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
        .TextFrame.AutoSize = ppAutoSizeNone
        .TextEffect.FontSize = font_size
    End With
End Sub

' �e�L�X�g�����w�u���b�N
Public Sub LayerBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    layer_w As Single, layer_h As Single)
    ' �����I�u�W�F�N�g�̉E�[����ɑw�u���b�N��u��
    '
    ' Args
    ' name: �w�u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' layer_w: �w�u���b�N�̕�
    ' layer_h: �w�u���b�N�̍���
    Dim layer As PowerPoint.Shape
    Set layer = Me.sld.Shapes.AddShape( _
        Type:=msoShapeRectangle, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - layer_h) / 2 + offset_y, _
        Width:=layer_w, _
        Height:=layer_h)
    With layer
        .Select
        .name = name
        .Fill.ForeColor.RGB = RGB(170, 207, 82)
        .Line.Visible = msoFalse
    End With
End Sub

' �e�L�X�g�t���w�u���b�N
Public Sub LayerAndTextBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    layer_w As Single, layer_h As Single, left_txt_w As Single, left_txt_h As Single, left_txt_content As String, _
    top_txt_w As Single, top_txt_h As Single, top_txt_content As String, txt_offset As Single, font_size As Single)
    ' �����I�u�W�F�N�g�̉E�[����ɑw�u���b�N��u��
    '
    ' Args
    ' name: �w�u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' layer_w: �w�u���b�N�̕�
    ' layer_h: �w�u���b�N�̍���
    ' left_txt_w: ���ɕt���e�L�X�g�{�b�N�X�̕�
    ' left_txt_h: ���ɕt���e�L�X�g�{�b�N�X�̍���
    ' left_txt_content: ���ɕt���e�L�X�g�̒��g
    ' top_txt_w: ��ɕt���e�L�X�g�{�b�N�X�̕�
    ' top_txt_h: ��ɕt���e�L�X�g�{�b�N�X�̍���
    ' top_txt_content: ��ɕt���e�L�X�g�̒��g
    ' txt_offset: �e�L�X�g�{�b�N�X�̑w�u���b�N����̃I�t�Z�b�g
    ' font_size: �t�H���g�T�C�Y
    Dim layer As PowerPoint.Shape
    Set layer = Me.sld.Shapes.AddShape( _
        Type:=msoShapeRectangle, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - layer_h) / 2 + offset_y, _
        Width:=layer_w, _
        Height:=layer_h)
    With layer
        .Select
        .name = name
        .Fill.ForeColor.RGB = RGB(170, 207, 82)
        .Line.Visible = msoFalse
    End With
    
    ' ���e�L�X�g�ǉ�
    Dim left_txt As PowerPoint.Shape
    Set left_txt = Me.sld.Shapes.AddTextbox( _
        orientation:=msoTextOrientationHorizontal, _
        Left:=layer.Left - (left_txt_w + left_txt_h) / 2 - txt_offset, _
        Top:=layer.Top + layer.Height - (left_txt_w + left_txt_h) / 2, _
        Width:=left_txt_w, _
        Height:=left_txt_h)
    With left_txt
        .Select
        .TextFrame.TextRange = left_txt_content
        .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
        .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
        .TextFrame.AutoSize = ppAutoSizeNone
        .TextEffect.FontSize = font_size
        .Rotation = -90
    End With
    
    ' ��e�L�X�g�ǉ�
    Dim top_txt As PowerPoint.Shape
    Set top_txt = Me.sld.Shapes.AddTextbox( _
        orientation:=msoTextOrientationHorizontal, _
        Left:=layer.Left + (layer.Width - top_txt_w) / 2, _
        Top:=layer.Top - top_txt_h - txt_offset, _
        Width:=top_txt_w, _
        Height:=top_txt_h)
    With top_txt
        .Select
        .TextFrame.TextRange = top_txt_content
        .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
        .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
        .TextFrame.AutoSize = ppAutoSizeNone
        .TextEffect.FontSize = font_size
    End With
End Sub

' �}�[�N�u���b�N
Public Sub MarkBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    mark_w As Single, mark_h As Single, mark_content As String, line_weight As Single, font_size As Single)
    ' �����I�u�W�F�N�g�̉E�[����Ƀ}�[�N�u���b�N��u��
    '
    ' Args
    ' name: �}�[�N�u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' mark_w: �}�[�N�u���b�N�̕�
    ' mark_h: �}�[�N�u���b�N�̍���
    ' mark_content: �}�[�N�̒��g
    ' line_weight: �g���̑���
    ' font_size: �t�H���g�T�C�Y
    Dim mark As PowerPoint.Shape
    Set mark = Me.sld.Shapes.AddShape( _
        Type:=msoShapeOval, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - mark_h) / 2 + offset_y, _
        Width:=mark_w, _
        Height:=mark_h)
    With mark
        .Select
        .name = name
        .Fill.Visible = msoFalse
        .Line.ForeColor.RGB = RGB(0, 0, 0)
        .Line.weight = line_weight
        .TextFrame.TextRange = mark_content
        .TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 0)
        .TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
        .TextFrame.AutoSize = ppAutoSizeNone
        .TextEffect.FontSize = font_size
    End With
End Sub

' ���u���b�N
Public Sub ArrowBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    arrow_w As Single, arrow_h As Single, orientation As String)
    ' �����I�u�W�F�N�g�̉E�[or���[or��[or���[�iorientation�ɂ��j����ɖ��u���b�N��u��
    '
    ' Args
    ' name: ���u���b�N�̖��O
    ' from_name: �I�t�Z�b�g�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' arrow_w: ���u���b�N�̕�
    ' arrow_h: ���u���b�N�̍���
    ' orientation: ���̌����B�i"Right"��"Left"��"Up"��"Down"�j
    Dim arrow As PowerPoint.Shape
    Select Case orientation
        Case "Right"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeRightArrow, _
                Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
                Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - arrow_h) / 2 + offset_y, _
                Width:=arrow_w, _
                Height:=arrow_h)
        Case "Left"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeLeftArrow, _
                Left:=Me.sld.Shapes(from_name).Left - arrow_w + offset_x, _
                Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - arrow_h) / 2 + offset_y, _
                Width:=arrow_w, _
                Height:=arrow_h)
        Case "Up"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeUpArrow, _
                Left:=Me.sld.Shapes(from_name).Left + (Me.sld.Shapes(from_name).Width - arrow_w) / 2 + offset_x, _
                Top:=Me.sld.Shapes(from_name).Top - arrow_h + offset_y, _
                Width:=arrow_w, _
                Height:=arrow_h)
        Case "Down"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeDownArrow, _
                Left:=Me.sld.Shapes(from_name).Left + (Me.sld.Shapes(from_name).Width - arrow_w) / 2 + offset_x, _
                Top:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height + offset_y, _
                Width:=arrow_w, _
                Height:=arrow_h)
    End Select
    With arrow
        .Select
        .name = name
        .Fill.ForeColor.RGB = RGB(0, 169, 95)
        .Line.Visible = msoFalse
    End With
End Sub

' ���u���b�N�R�l�N�^
Public Sub ArrowBlockConnector(name As String, from_name As String, to_name As String, _
    offset_x As Single, offset_y As Single, arrow_weight As Single, orientation As String)
    ' �����I�u�W�F�N�g�̉E�[or���[or��[or���[�iorientation�ɂ��j������u���b�N�R�l�N�^��u��
    '
    ' Args
    ' name: ���u���b�N�R�l�N�^�̖��O
    ' from_name: �J�n�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' to_name: �I���̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' offset_x: X�����̃I�t�Z�b�g
    ' offset_y: Y�����̃I�t�Z�b�g
    ' arrow_weight: ���̑���
    ' orientation: ���̌����B�i"Right"��"Left"��"Up"��"Down"�j
    Dim arrow As PowerPoint.Shape
    Select Case orientation
        Case "Right"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeRightArrow, _
                Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
                Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - arrow_weight) / 2 + offset_y, _
                Width:=Me.sld.Shapes(to_name).Left - Me.sld.Shapes(from_name).Left - Me.sld.Shapes(from_name).Width - 2 * offset_x, _
                Height:=arrow_weight)
        Case "Left"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeLeftArrow, _
                Left:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width - offset_x, _
                Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - arrow_weight) / 2 + offset_y, _
                Width:=Me.sld.Shapes(from_name).Left - Me.sld.Shapes(to_name).Left - Me.sld.Shapes(to_name).Width + 2 * offset_x, _
                Height:=arrow_weight)
        Case "Up"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeUpArrow, _
                Left:=Me.sld.Shapes(from_name).Left + (Me.sld.Shapes(from_name).Width - arrow_weight) / 2 + offset_x, _
                Top:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height - offset_y, _
                Width:=arrow_weight, _
                Height:=Me.sld.Shapes(from_name).Top - Me.sld.Shapes(to_name).Top - Me.sld.Shapes(to_name).Height + 2 * offset_y)
        Case "Down"
            Set arrow = Me.sld.Shapes.AddShape( _
                Type:=msoShapeDownArrow, _
                Left:=Me.sld.Shapes(from_name).Left + (Me.sld.Shapes(from_name).Width - arrow_weight) / 2 + offset_x, _
                Top:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height + offset_y, _
                Width:=arrow_weight, _
                Height:=Me.sld.Shapes(to_name).Top - Me.sld.Shapes(from_name).Top - Me.sld.Shapes(from_name).Height - 2 * offset_y)
    End Select
    With arrow
        .Select
        .name = name
        .Fill.ForeColor.RGB = RGB(165, 165, 165)
        .Line.Visible = msoFalse
    End With
End Sub

' �����R�l�N�^
Public Sub ArrowLineConnector(name As String, from_name As String, to_name As String, _
    line_weight As Single, orientation As String)
    ' �����I�u�W�F�N�g�̉E�[or���[or��[or���[�iorientation�ɂ��j��������R�l�N�^��u��
    '
    ' Args
    ' name: �����R�l�N�^�̖��O
    ' from_name: �J�n�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' to_name: �I���̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' line_weight: ���̑���
    ' orientation: ���̌����B�i"Right"��"Left"��"Up"��"Down"�j
    Dim arrow As PowerPoint.Shape
    Select Case orientation
        Case "Right"
            Set arrow = Me.sld.Shapes.AddLine( _
                BeginX:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width, _
                BeginY:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height / 2, _
                EndX:=Me.sld.Shapes(to_name).Left, _
                EndY:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height / 2)
        Case "Left"
            Set arrow = Me.sld.Shapes.AddLine( _
                BeginX:=Me.sld.Shapes(from_name).Left, _
                BeginY:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height / 2, _
                EndX:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width, _
                EndY:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height / 2)
        Case "Up"
            Set arrow = Me.sld.Shapes.AddLine( _
                BeginX:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width / 2, _
                BeginY:=Me.sld.Shapes(from_name).Top, _
                EndX:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width / 2, _
                EndY:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height)
        Case "Down"
            Set arrow = Me.sld.Shapes.AddLine( _
                BeginX:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width / 2, _
                BeginY:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height, _
                EndX:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width / 2, _
                EndY:=Me.sld.Shapes(to_name).Top)
    End Select
    With arrow
        .Select
        .name = name
        .Line.ForeColor.RGB = RGB(0, 0, 0)
        .Line.weight = line_weight
        .Line.EndArrowheadStyle = msoArrowheadTriangle
    End With
End Sub

' ��������R�l�N�^
Public Sub BranchArrowLineConnector(name As String, from_name As String, to_name As String, line_weight As Single)
    ' �����I�u�W�F�N�g�̉��[����ʂ̃I�u�W�F�N�g�̍��[�֕�������R�l�N�^��u��
    '
    ' Args
    ' name: ��������R�l�N�^�̖��O
    ' from_name: �J�n�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' to_name: �I���̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' line_weight: ���̑���
    Dim arrow As PowerPoint.FreeformBuilder
    Set arrow = Me.sld.Shapes.BuildFreeform( _
        EditingType:=msoEditingCorner, _
        X1:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width / 2, _
        Y1:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height)
    With arrow
        .AddNodes SegmentType:=msoSegmentLine, EditingType:=msoEditingCorner, _
            X1:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width / 2, _
            Y1:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height / 2
        .AddNodes SegmentType:=msoSegmentLine, EditingType:=msoEditingCorner, _
            X1:=Me.sld.Shapes(to_name).Left, _
            Y1:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height / 2
        With .ConvertToShape
            .Select
            .name = name
            .Line.ForeColor.RGB = RGB(0, 0, 0)
            .Line.weight = line_weight
            .Line.EndArrowheadStyle = msoArrowheadTriangle
        End With
    End With
End Sub

' ���������R�l�N�^
Public Sub MergeArrowLineConnector(name As String, from_name As String, to_name As String, line_weight As Single)
    ' �����I�u�W�F�N�g�̉E�[����ʂ̃I�u�W�F�N�g�̉��[�֍��������R�l�N�^��u��
    '
    ' Args
    ' name: ���������R�l�N�^�̖��O
    ' from_name: �J�n�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' to_name: �I���̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' line_weight: ���̑���
    Dim arrow As PowerPoint.FreeformBuilder
    Set arrow = Me.sld.Shapes.BuildFreeform( _
        EditingType:=msoEditingCorner, _
        X1:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width, _
        Y1:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height / 2)
    With arrow
        .AddNodes SegmentType:=msoSegmentLine, EditingType:=msoEditingCorner, _
            X1:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width / 2, _
            Y1:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height / 2
        .AddNodes SegmentType:=msoSegmentLine, EditingType:=msoEditingCorner, _
            X1:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width / 2, _
            Y1:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height
        With .ConvertToShape
            .Select
            .name = name
            .Line.ForeColor.RGB = RGB(0, 0, 0)
            .Line.weight = line_weight
            .Line.EndArrowheadStyle = msoArrowheadTriangle
        End With
    End With
End Sub

' �X�L�b�v�����R�l�N�^
Public Sub SkipArrowLineConnector(name As String, from_name As String, to_name As String, line_h As Single, line_weight As Single)
    ' �����I�u�W�F�N�g�̉��[����ʂ̃I�u�W�F�N�g�̉��[�փX�L�b�v�����R�l�N�^��u��
    '
    ' Args
    ' name: �X�L�b�v�����R�l�N�^�̖��O
    ' from_name: �J�n�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' to_name: �I���̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' line_h: �X�L�b�v������from_name����̍����I�t�Z�b�g
    ' line_weight: ���̑���
    Dim arrow As PowerPoint.FreeformBuilder
    Set arrow = Me.sld.Shapes.BuildFreeform( _
        EditingType:=msoEditingCorner, _
        X1:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width / 2, _
        Y1:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height)
    With arrow
        .AddNodes SegmentType:=msoSegmentLine, EditingType:=msoEditingCorner, _
            X1:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width / 2, _
            Y1:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height + line_h
        .AddNodes SegmentType:=msoSegmentLine, EditingType:=msoEditingCorner, _
            X1:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width / 2, _
            Y1:=Me.sld.Shapes(from_name).Top + Me.sld.Shapes(from_name).Height + line_h
        .AddNodes SegmentType:=msoSegmentLine, EditingType:=msoEditingCorner, _
            X1:=Me.sld.Shapes(to_name).Left + Me.sld.Shapes(to_name).Width / 2, _
            Y1:=Me.sld.Shapes(to_name).Top + Me.sld.Shapes(to_name).Height
        With .ConvertToShape
            .Select
            .name = name
            .Line.ForeColor.RGB = RGB(0, 0, 0)
            .Line.weight = line_weight
            .Line.EndArrowheadStyle = msoArrowheadTriangle
        End With
    End With
End Sub

' �J�M�����R�l�N�^
Public Sub ElbowArrowLineConnector(name As String, from_name As String, to_name As String, line_weight As Single)
    ' �����I�u�W�F�N�g�̏�[����ʂ̃I�u�W�F�N�g�̉��[�փJ�M�����R�l�N�^��u��
    '
    ' Args
    ' name: �J�M�����R�l�N�^�̖��O
    ' from_name: �J�n�̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' to_name: �I���̊�ƂȂ�I�u�W�F�N�g�̖��O
    ' line_weight: ���̑���
    Dim arrow As PowerPoint.Shape
    Set arrow = Me.sld.Shapes.AddConnector(msoConnectorElbow, 0, 0, 0, 0)
    With arrow
        .Select
        .name = name
        With .ConnectorFormat
            .BeginConnect Me.sld.Shapes(from_name), 1
            .EndConnect Me.sld.Shapes(to_name), 3
        End With
        .Line.ForeColor.RGB = RGB(0, 0, 0)
        .Line.weight = line_weight
        .Line.EndArrowheadStyle = msoArrowheadTriangle
    End With
End Sub
