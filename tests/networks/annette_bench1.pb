
b
xPlaceholder*
shape:*
dtype0*&
_output_shapes
:
l
conv2d/VariableConst*
dtype0*E
value<B:"$?N<`??<ph?<????𯼉B???Y.<s?j?5s<<
?
conv2d/Variable/readIdentityconv2d/Variable*&
_output_shapes
:*
T0*"
_class
loc:@conv2d/Variable
?
conv2d/Conv2DConv2Dxconv2d/Variable/read*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
J
conv2d/Variable_1Const*
dtype0*!
valueB"            
?
conv2d/Variable_1/readIdentityconv2d/Variable_1*
T0*$
_class
loc:@conv2d/Variable_1*
_output_shapes
:
?
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/Variable_1/read*
T0*
data_formatNHWC*&
_output_shapes
:
T
conv2d/ReluReluconv2d/BiasAdd*
T0*&
_output_shapes
:
n
conv2d_1/VariableConst*
dtype0*E
value<B:"$\Т=C=?=?E?[?ԼPq?=-}y=?7=C?<
?
conv2d_1/Variable/readIdentityconv2d_1/Variable*&
_output_shapes
:*
T0*$
_class
loc:@conv2d_1/Variable
?
conv2d_1/Conv2DConv2Dxconv2d_1/Variable/read*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
L
conv2d_1/Variable_1Const*!
valueB"            *
dtype0
?
conv2d_1/Variable_1/readIdentityconv2d_1/Variable_1*
T0*&
_class
loc:@conv2d_1/Variable_1*
_output_shapes
:
?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/Variable_1/read*
T0*
data_formatNHWC*&
_output_shapes
:
X
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*&
_output_shapes
:
n
conv2d_2/VariableConst*E
value<B:"$?v??G?<ݼ6?O?
=?	<Y?)??IK?^??R?=*
dtype0
?
conv2d_2/Variable/readIdentityconv2d_2/Variable*&
_output_shapes
:*
T0*$
_class
loc:@conv2d_2/Variable
?
conv2d_2/Conv2DConv2Dxconv2d_2/Variable/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*&
_output_shapes
:*
	dilations

L
conv2d_2/Variable_1Const*!
valueB"            *
dtype0
?
conv2d_2/Variable_1/readIdentityconv2d_2/Variable_1*
T0*&
_class
loc:@conv2d_2/Variable_1*
_output_shapes
:
?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/Variable_1/read*
T0*
data_formatNHWC*&
_output_shapes
:
X
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*&
_output_shapes
:
W
AddAddconv2d/Reluconv2d_1/Relu*
T0*&
_output_shapes
:
Q
Add_1AddAddconv2d_2/Relu*
T0*&
_output_shapes
:
f
!separable_conv2d/depthwise_kernelConst*
dtype0*-
value$B""X?ؾ??????I?
?
&separable_conv2d/depthwise_kernel/readIdentity!separable_conv2d/depthwise_kernel*&
_output_shapes
:*
T0*4
_class*
(&loc:@separable_conv2d/depthwise_kernel
~
!separable_conv2d/pointwise_kernelConst*E
value<B:"$ ??;0T?>?,q?`???(?
??\i???X?>*
dtype0
?
&separable_conv2d/pointwise_kernel/readIdentity!separable_conv2d/pointwise_kernel*&
_output_shapes
:*
T0*4
_class*
(&loc:@separable_conv2d/pointwise_kernel
N
separable_conv2d/biasConst*!
valueB"            *
dtype0
?
separable_conv2d/bias/readIdentityseparable_conv2d/bias*
_output_shapes
:*
T0*(
_class
loc:@separable_conv2d/bias
?
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeAdd_1&separable_conv2d/depthwise_kernel/read*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides

?
!separable_conv2d/separable_conv2dConv2D+separable_conv2d/separable_conv2d/depthwise&separable_conv2d/pointwise_kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*&
_output_shapes
:*
	dilations

?
separable_conv2d/BiasAddBiasAdd!separable_conv2d/separable_conv2dseparable_conv2d/bias/read*
T0*
data_formatNHWC*&
_output_shapes
:
n
conv2d_3/VariableConst*E
value<B:"$??,=뮒=?? ?⩸??<Z=M?=\??L?<??#?*
dtype0
?
conv2d_3/Variable/readIdentityconv2d_3/Variable*
T0*$
_class
loc:@conv2d_3/Variable*&
_output_shapes
:
?
conv2d_3/Conv2DConv2Dseparable_conv2d/BiasAddconv2d_3/Variable/read*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
L
conv2d_3/Variable_1Const*!
valueB"            *
dtype0
?
conv2d_3/Variable_1/readIdentityconv2d_3/Variable_1*
T0*&
_class
loc:@conv2d_3/Variable_1*
_output_shapes
:
?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/Variable_1/read*
T0*
data_formatNHWC*&
_output_shapes
:
X
conv2d_3/ReluReluconv2d_3/BiasAdd*
T0*&
_output_shapes
:
n
conv2d_4/VariableConst*E
value<B:"$?c?M??<?W?=]??:?????????< ն=?j??*
dtype0
?
conv2d_4/Variable/readIdentityconv2d_4/Variable*
T0*$
_class
loc:@conv2d_4/Variable*&
_output_shapes
:
?
conv2d_4/Conv2DConv2Dseparable_conv2d/BiasAddconv2d_4/Variable/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:
L
conv2d_4/Variable_1Const*!
valueB"            *
dtype0
?
conv2d_4/Variable_1/readIdentityconv2d_4/Variable_1*
_output_shapes
:*
T0*&
_class
loc:@conv2d_4/Variable_1
?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/Variable_1/read*&
_output_shapes
:*
T0*
data_formatNHWC
X
conv2d_4/ReluReluconv2d_4/BiasAdd*
T0*&
_output_shapes
:
?
max_pool/MaxPoolMaxPoolconv2d_4/Relu*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*&
_output_shapes
:*
T0
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?
concatConcatV2conv2d_3/Relumax_pool/MaxPoolconcat/axis*
T0*
N*&
_output_shapes
:*

Tidx0
f
flatten/Reshape/shapeConst*
_output_shapes
:*
valueB"????   *
dtype0
p
flatten/ReshapeReshapeconcatflatten/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
?
fully_conn/VariableConst*a
valueXBV"H?h???n???I??E??v?=??=?,߼??<+?i??;=l?B=5ڃ??Q?yQ?=,O?%?t?R???*
dtype0
?
fully_conn/Variable/readIdentityfully_conn/Variable*
T0*&
_class
loc:@fully_conn/Variable*
_output_shapes

:
N
fully_conn/Variable_1Const*
dtype0*!
valueB"            
?
fully_conn/Variable_1/readIdentityfully_conn/Variable_1*
T0*(
_class
loc:@fully_conn/Variable_1*
_output_shapes
:
?
fully_conn/MatMulMatMulflatten/Reshapefully_conn/Variable/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b( 
m
fully_conn/AddAddfully_conn/MatMulfully_conn/Variable_1/read*
T0*
_output_shapes

:
P
fully_conn/ReluRelufully_conn/Add*
T0*
_output_shapes

:
j
fully_conn_1/VariableConst*=
value4B2"$s.????z?ۼj??<?=?;sZѼ????e=
ŋ?*
dtype0
?
fully_conn_1/Variable/readIdentityfully_conn_1/Variable*(
_class
loc:@fully_conn_1/Variable*
_output_shapes

:*
T0
P
fully_conn_1/Variable_1Const*!
valueB"            *
dtype0
?
fully_conn_1/Variable_1/readIdentityfully_conn_1/Variable_1*
T0**
_class 
loc:@fully_conn_1/Variable_1*
_output_shapes
:
?
fully_conn_1/MatMulMatMulfully_conn/Relufully_conn_1/Variable/read*
_output_shapes

:*
transpose_a( *
transpose_b( *
T0
s
fully_conn_1/AddAddfully_conn_1/MatMulfully_conn_1/Variable_1/read*
_output_shapes

:*
T0
Z
fully_conn_1/SoftmaxSoftmaxfully_conn_1/Add*
T0*
_output_shapes

: 