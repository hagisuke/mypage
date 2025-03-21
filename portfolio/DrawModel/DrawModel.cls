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

' メンバ変数
Public prs As PowerPoint.Presentation
Public sld As PowerPoint.Slide

' 初期化処理
Private Sub Class_Initialize()
    ' 開いているプレゼンテーションをprsに格納
    Set Me.prs = ActivePresentation
    ' 白紙スライド追加
    Set Me.sld = Me.prs.Slides.Add(Me.prs.Slides.Count + 1, ppLayoutBlank)
    Me.sld.Select
    Debug.Print "initalized"
End Sub

' 終了時処理
Private Sub Class_Terminate()
    ActiveWindow.Selection.Unselect
    Debug.Print "terminated"
End Sub

' 開始ブロック
Public Sub StartBlock(name As String, start_w As Single, start_h As Single)
    ' スライドの左端・真ん中の高さに透明なブロックを置く
    '
    ' Args
    ' name: 開始ブロックの名前
    ' start_w: 開始ブロックの幅
    ' start_h: 開始ブロックの高さ
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

' テキストブロック
Public Sub TextBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    txt_w As Single, txt_h As Single, txt_content As String, font_size As Single)
    ' 既存オブジェクトの右端を基準にテキストボックスを置く
    '
    ' Args
    ' name: テキストブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' txt_w: テキストブロックの幅
    ' txt_h: テキストブロックの高さ
    ' txt_content: テキストの中身
    ' font_size: フォントサイズ
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

' 画像ブロック
Public Sub ImageBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    img_w As Single, img_h As Single, img_path As String, Optional img_aspect As MsoTriState = msoTrue)
    ' 既存オブジェクトの右端を基準に画像を置く
    '
    ' Args
    ' name: 画像ブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' img_w: 画像の幅
    ' img_h: 画像の高さ
    ' img_path: 画像ファイルへのパス
    ' img_aspect: 縦横比固定のフラグ（img_hに合わせる。Default: msoTrue）
    Dim img As PowerPoint.Shape
    Set img = Me.sld.Shapes.AddPicture(FileName:=img_path, _
        LinkToFile:=msoFalse, _
        SaveWithDocument:=msoTrue, _
        Left:=Me.sld.Shapes(from_name).Left + Me.sld.Shapes(from_name).Width + offset_x, _
        Top:=Me.sld.Shapes(from_name).Top + (Me.sld.Shapes(from_name).Height - img_h) / 2 + offset_y)
    With img
        .LockAspectRatio = img_aspect ' 縦横比を固定
        .Select
        .name = name
        ' 画像サイズ変更
        .Width = img_w
        .Height = img_h
    End With
End Sub

' キューブブロック
Public Sub CubeBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    cube_w As Single, cube_h As Single, cube_d As Single, txt_content As String, font_size As Single, Optional depth_mode As MsoTriState = msoFalse)
    ' 既存オブジェクトの右端を基準に立方体（直方体）を置く
    '
    ' Args
    ' name: キューブブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' cube_w: キューブブロックの幅
    ' cube_h: キューブブロックの高さ
    ' cube_d: キューブブロックの奥行き
    ' txt_content: テキストの中身
    ' font_size: フォントサイズ
    ' depth_mode: msoTrueならキューブを幅・高さ・奥行きで指定し、msoFalseならオブジェクトの幅・高さで指定（Default: msoFalse）
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

' 処理ブロック
Public Sub ProcessBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    process_w As Single, process_h As Single, txt_content As String, font_size As Single, txt_orientation As String)
    ' 既存オブジェクトの右端を基準に処理ブロックを置く
    '
    ' Args
    ' name: 処理ブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' process_w: 処理ブロックの幅
    ' process_h: 処理ブロックの高さ
    ' txt_content: テキストの中身
    ' font_size: フォントサイズ
    ' txt_orientation: テキストの方向（"h"か"v"）
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

' ダウンサンプルブロック
Public Sub DownsampleBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    process_w As Single, process_h As Single, txt_content As String, font_size As Single, orientation As String)
    ' 既存オブジェクトの右端を基準にダウンサンプルブロックを置く
    '
    ' Args
    ' name: ダウンサンプルブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' process_w: ダウンサンプルブロックの幅
    ' process_h: ダウンサンプルブロックの高さ
    ' txt_content: テキストの中身
    ' font_size: フォントサイズ
    ' orientation: オブジェクトの方向（"h"か"v"）
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
    
    ' 座標系維持のためのベース追加
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

' アップサンプルブロック
Public Sub UpsampleBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    process_w As Single, process_h As Single, txt_content As String, font_size As Single, orientation As String)
    ' 既存オブジェクトの右端を基準にアップサンプルブロックを置く
    '
    ' Args
    ' name: アップサンプルブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' process_w: アップサンプルブロックの幅
    ' process_h: アップサンプルブロックの高さ
    ' txt_content: テキストの中身
    ' font_size: フォントサイズ
    ' orientation: オブジェクトの方向（"h"か"v"）
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
    
    ' 座標系維持のためのベース追加
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

' テキスト無し層ブロック
Public Sub LayerBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    layer_w As Single, layer_h As Single)
    ' 既存オブジェクトの右端を基準に層ブロックを置く
    '
    ' Args
    ' name: 層ブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' layer_w: 層ブロックの幅
    ' layer_h: 層ブロックの高さ
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

' テキスト付き層ブロック
Public Sub LayerAndTextBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    layer_w As Single, layer_h As Single, left_txt_w As Single, left_txt_h As Single, left_txt_content As String, _
    top_txt_w As Single, top_txt_h As Single, top_txt_content As String, txt_offset As Single, font_size As Single)
    ' 既存オブジェクトの右端を基準に層ブロックを置く
    '
    ' Args
    ' name: 層ブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' layer_w: 層ブロックの幅
    ' layer_h: 層ブロックの高さ
    ' left_txt_w: 左に付くテキストボックスの幅
    ' left_txt_h: 左に付くテキストボックスの高さ
    ' left_txt_content: 左に付くテキストの中身
    ' top_txt_w: 上に付くテキストボックスの幅
    ' top_txt_h: 上に付くテキストボックスの高さ
    ' top_txt_content: 上に付くテキストの中身
    ' txt_offset: テキストボックスの層ブロックからのオフセット
    ' font_size: フォントサイズ
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
    
    ' 左テキスト追加
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
    
    ' 上テキスト追加
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

' マークブロック
Public Sub MarkBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    mark_w As Single, mark_h As Single, mark_content As String, line_weight As Single, font_size As Single)
    ' 既存オブジェクトの右端を基準にマークブロックを置く
    '
    ' Args
    ' name: マークブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' mark_w: マークブロックの幅
    ' mark_h: マークブロックの高さ
    ' mark_content: マークの中身
    ' line_weight: 枠線の太さ
    ' font_size: フォントサイズ
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

' 矢印ブロック
Public Sub ArrowBlock(name As String, from_name As String, offset_x As Single, offset_y As Single, _
    arrow_w As Single, arrow_h As Single, orientation As String)
    ' 既存オブジェクトの右端or左端or上端or下端（orientationによる）を基準に矢印ブロックを置く
    '
    ' Args
    ' name: 矢印ブロックの名前
    ' from_name: オフセットの基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' arrow_w: 矢印ブロックの幅
    ' arrow_h: 矢印ブロックの高さ
    ' orientation: 矢印の向き。（"Right"か"Left"か"Up"か"Down"）
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

' 矢印ブロックコネクタ
Public Sub ArrowBlockConnector(name As String, from_name As String, to_name As String, _
    offset_x As Single, offset_y As Single, arrow_weight As Single, orientation As String)
    ' 既存オブジェクトの右端or左端or上端or下端（orientationによる）から矢印ブロックコネクタを置く
    '
    ' Args
    ' name: 矢印ブロックコネクタの名前
    ' from_name: 開始の基準となるオブジェクトの名前
    ' to_name: 終了の基準となるオブジェクトの名前
    ' offset_x: X方向のオフセット
    ' offset_y: Y方向のオフセット
    ' arrow_weight: 矢印の太さ
    ' orientation: 矢印の向き。（"Right"か"Left"か"Up"か"Down"）
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

' 矢印線コネクタ
Public Sub ArrowLineConnector(name As String, from_name As String, to_name As String, _
    line_weight As Single, orientation As String)
    ' 既存オブジェクトの右端or左端or上端or下端（orientationによる）から矢印線コネクタを置く
    '
    ' Args
    ' name: 矢印線コネクタの名前
    ' from_name: 開始の基準となるオブジェクトの名前
    ' to_name: 終了の基準となるオブジェクトの名前
    ' line_weight: 矢印の太さ
    ' orientation: 矢印の向き。（"Right"か"Left"か"Up"か"Down"）
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

' 分岐矢印線コネクタ
Public Sub BranchArrowLineConnector(name As String, from_name As String, to_name As String, line_weight As Single)
    ' 既存オブジェクトの下端から別のオブジェクトの左端へ分岐矢印線コネクタを置く
    '
    ' Args
    ' name: 分岐矢印線コネクタの名前
    ' from_name: 開始の基準となるオブジェクトの名前
    ' to_name: 終了の基準となるオブジェクトの名前
    ' line_weight: 矢印の太さ
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

' 合流矢印線コネクタ
Public Sub MergeArrowLineConnector(name As String, from_name As String, to_name As String, line_weight As Single)
    ' 既存オブジェクトの右端から別のオブジェクトの下端へ合流矢印線コネクタを置く
    '
    ' Args
    ' name: 合流矢印線コネクタの名前
    ' from_name: 開始の基準となるオブジェクトの名前
    ' to_name: 終了の基準となるオブジェクトの名前
    ' line_weight: 矢印の太さ
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

' スキップ矢印線コネクタ
Public Sub SkipArrowLineConnector(name As String, from_name As String, to_name As String, line_h As Single, line_weight As Single)
    ' 既存オブジェクトの下端から別のオブジェクトの下端へスキップ矢印線コネクタを置く
    '
    ' Args
    ' name: スキップ矢印線コネクタの名前
    ' from_name: 開始の基準となるオブジェクトの名前
    ' to_name: 終了の基準となるオブジェクトの名前
    ' line_h: スキップ矢印線のfrom_nameからの高さオフセット
    ' line_weight: 矢印の太さ
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

' カギ矢印線コネクタ
Public Sub ElbowArrowLineConnector(name As String, from_name As String, to_name As String, line_weight As Single)
    ' 既存オブジェクトの上端から別のオブジェクトの下端へカギ矢印線コネクタを置く
    '
    ' Args
    ' name: カギ矢印線コネクタの名前
    ' from_name: 開始の基準となるオブジェクトの名前
    ' to_name: 終了の基準となるオブジェクトの名前
    ' line_weight: 矢印の太さ
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

