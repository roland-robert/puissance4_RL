¬ñ
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02unknown8°¸

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
:*
dtype0
u
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_17/bias
n
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes	
:*
dtype0
|
dense_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_93/kernel
u
#dense_93/kernel/Read/ReadVariableOpReadVariableOpdense_93/kernel* 
_output_shapes
:
*
dtype0
s
dense_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_93/bias
l
!dense_93/bias/Read/ReadVariableOpReadVariableOpdense_93/bias*
_output_shapes	
:*
dtype0
|
dense_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_94/kernel
u
#dense_94/kernel/Read/ReadVariableOpReadVariableOpdense_94/kernel* 
_output_shapes
:
*
dtype0
s
dense_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_94/bias
l
!dense_94/bias/Read/ReadVariableOpReadVariableOpdense_94/bias*
_output_shapes	
:*
dtype0
{
dense_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_95/kernel
t
#dense_95/kernel/Read/ReadVariableOpReadVariableOpdense_95/kernel*
_output_shapes
:	*
dtype0
r
dense_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_95/bias
k
!dense_95/bias/Read/ReadVariableOpReadVariableOpdense_95/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
Æ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value÷Bô Bí
§
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
6
)iter
	*decay
+learning_rate
,momentum
 
8
0
1
2
3
4
5
#6
$7
8
0
1
2
3
4
5
#6
$7

-metrics
.non_trainable_variables
regularization_losses
		variables

trainable_variables
/layer_regularization_losses

0layers
 
\Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

1metrics
2non_trainable_variables
regularization_losses
	variables
trainable_variables
3layer_regularization_losses

4layers
 
 
 

5metrics
6non_trainable_variables
regularization_losses
	variables
trainable_variables
7layer_regularization_losses

8layers
[Y
VARIABLE_VALUEdense_93/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_93/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

9metrics
:non_trainable_variables
regularization_losses
	variables
trainable_variables
;layer_regularization_losses

<layers
[Y
VARIABLE_VALUEdense_94/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_94/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

=metrics
>non_trainable_variables
regularization_losses
 	variables
!trainable_variables
?layer_regularization_losses

@layers
[Y
VARIABLE_VALUEdense_95/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_95/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1

Ametrics
Bnon_trainable_variables
%regularization_losses
&	variables
'trainable_variables
Clayer_regularization_losses

Dlayers
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

E0
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	Ftotal
	Gcount
H
_fn_kwargs
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

F0
G1
 

Mmetrics
Nnon_trainable_variables
Iregularization_losses
J	variables
Ktrainable_variables
Olayer_regularization_losses

Players
 

F0
G1
 
 

serving_default_conv2d_17_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
®
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_17_inputconv2d_17/kernelconv2d_17/biasdense_93/kerneldense_93/biasdense_94/kerneldense_94/biasdense_95/kerneldense_95/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*/
f*R(
&__inference_signature_wrapper_66749899
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ó
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp#dense_93/kernel/Read/ReadVariableOp!dense_93/bias/Read/ReadVariableOp#dense_94/kernel/Read/ReadVariableOp!dense_94/bias/Read/ReadVariableOp#dense_95/kernel/Read/ReadVariableOp!dense_95/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_save_66750116
Ö
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_17/kernelconv2d_17/biasdense_93/kerneldense_93/biasdense_94/kerneldense_94/biasdense_95/kerneldense_95/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference__traced_restore_66750170·ó
æ
I
-__inference_flatten_17_layer_call_fn_66749998

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_17_layer_call_and_return_conditional_losses_667497252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
¾


0__inference_sequential_31_layer_call_fn_66749987

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_31_layer_call_and_return_conditional_losses_667498682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
î&
æ
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749930

inputs,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource+
'dense_93_matmul_readvariableop_resource,
(dense_93_biasadd_readvariableop_resource+
'dense_94_matmul_readvariableop_resource,
(dense_94_biasadd_readvariableop_resource+
'dense_95_matmul_readvariableop_resource,
(dense_95_biasadd_readvariableop_resource
identity¢ conv2d_17/BiasAdd/ReadVariableOp¢conv2d_17/Conv2D/ReadVariableOp¢dense_93/BiasAdd/ReadVariableOp¢dense_93/MatMul/ReadVariableOp¢dense_94/BiasAdd/ReadVariableOp¢dense_94/MatMul/ReadVariableOp¢dense_95/BiasAdd/ReadVariableOp¢dense_95/MatMul/ReadVariableOp´
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_17/Conv2D/ReadVariableOpÃ
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_17/Conv2D«
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp±
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_17/BiasAddu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_17/Const
flatten_17/ReshapeReshapeconv2d_17/BiasAdd:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_17/Reshapeª
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_93/MatMul/ReadVariableOp¤
dense_93/MatMulMatMulflatten_17/Reshape:output:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/MatMul¨
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_93/BiasAdd/ReadVariableOp¦
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/BiasAddt
dense_93/TanhTanhdense_93/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/Tanhª
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_94/MatMul/ReadVariableOp
dense_94/MatMulMatMuldense_93/Tanh:y:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/MatMul¨
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_94/BiasAdd/ReadVariableOp¦
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/BiasAdd©
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_95/MatMul/ReadVariableOp¡
dense_95/MatMulMatMuldense_94/BiasAdd:output:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/MatMul§
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp¥
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/BiasAddû
IdentityIdentitydense_95/BiasAdd:output:0!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
û

à
G__inference_conv2d_17_layer_call_and_return_conditional_losses_66749704

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd°
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
ò
ß
F__inference_dense_94_layer_call_and_return_conditional_losses_66749766

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ù


0__inference_sequential_31_layer_call_fn_66749879
conv2d_17_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_31_layer_call_and_return_conditional_losses_667498682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_17_input
ò
ß
F__inference_dense_94_layer_call_and_return_conditional_losses_66750026

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ù
¬
+__inference_dense_94_layer_call_fn_66750033

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_94_layer_call_and_return_conditional_losses_667497662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ú
ê
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749801
conv2d_17_input,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2+
'dense_93_statefulpartitionedcall_args_1+
'dense_93_statefulpartitionedcall_args_2+
'dense_94_statefulpartitionedcall_args_1+
'dense_94_statefulpartitionedcall_args_2+
'dense_95_statefulpartitionedcall_args_1+
'dense_95_statefulpartitionedcall_args_2
identity¢!conv2d_17/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallÃ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_input(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_17_layer_call_and_return_conditional_losses_667497042#
!conv2d_17/StatefulPartitionedCallë
flatten_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_17_layer_call_and_return_conditional_losses_667497252
flatten_17/PartitionedCallÊ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0'dense_93_statefulpartitionedcall_args_1'dense_93_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_93_layer_call_and_return_conditional_losses_667497442"
 dense_93/StatefulPartitionedCallÐ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0'dense_94_statefulpartitionedcall_args_1'dense_94_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_94_layer_call_and_return_conditional_losses_667497662"
 dense_94/StatefulPartitionedCallÏ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0'dense_95_statefulpartitionedcall_args_1'dense_95_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_95_layer_call_and_return_conditional_losses_667497882"
 dense_95/StatefulPartitionedCall
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_17_input
Å	
ß
F__inference_dense_93_layer_call_and_return_conditional_losses_66750009

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
î&
æ
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749961

inputs,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource+
'dense_93_matmul_readvariableop_resource,
(dense_93_biasadd_readvariableop_resource+
'dense_94_matmul_readvariableop_resource,
(dense_94_biasadd_readvariableop_resource+
'dense_95_matmul_readvariableop_resource,
(dense_95_biasadd_readvariableop_resource
identity¢ conv2d_17/BiasAdd/ReadVariableOp¢conv2d_17/Conv2D/ReadVariableOp¢dense_93/BiasAdd/ReadVariableOp¢dense_93/MatMul/ReadVariableOp¢dense_94/BiasAdd/ReadVariableOp¢dense_94/MatMul/ReadVariableOp¢dense_95/BiasAdd/ReadVariableOp¢dense_95/MatMul/ReadVariableOp´
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_17/Conv2D/ReadVariableOpÃ
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_17/Conv2D«
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp±
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_17/BiasAddu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_17/Const
flatten_17/ReshapeReshapeconv2d_17/BiasAdd:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_17/Reshapeª
dense_93/MatMul/ReadVariableOpReadVariableOp'dense_93_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_93/MatMul/ReadVariableOp¤
dense_93/MatMulMatMulflatten_17/Reshape:output:0&dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/MatMul¨
dense_93/BiasAdd/ReadVariableOpReadVariableOp(dense_93_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_93/BiasAdd/ReadVariableOp¦
dense_93/BiasAddBiasAdddense_93/MatMul:product:0'dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/BiasAddt
dense_93/TanhTanhdense_93/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_93/Tanhª
dense_94/MatMul/ReadVariableOpReadVariableOp'dense_94_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_94/MatMul/ReadVariableOp
dense_94/MatMulMatMuldense_93/Tanh:y:0&dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/MatMul¨
dense_94/BiasAdd/ReadVariableOpReadVariableOp(dense_94_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_94/BiasAdd/ReadVariableOp¦
dense_94/BiasAddBiasAdddense_94/MatMul:product:0'dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_94/BiasAdd©
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_95/MatMul/ReadVariableOp¡
dense_95/MatMulMatMuldense_94/BiasAdd:output:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/MatMul§
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp¥
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_95/BiasAddû
IdentityIdentitydense_95/BiasAdd:output:0!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_93/BiasAdd/ReadVariableOp^dense_93/MatMul/ReadVariableOp ^dense_94/BiasAdd/ReadVariableOp^dense_94/MatMul/ReadVariableOp ^dense_95/BiasAdd/ReadVariableOp^dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_93/BiasAdd/ReadVariableOpdense_93/BiasAdd/ReadVariableOp2@
dense_93/MatMul/ReadVariableOpdense_93/MatMul/ReadVariableOp2B
dense_94/BiasAdd/ReadVariableOpdense_94/BiasAdd/ReadVariableOp2@
dense_94/MatMul/ReadVariableOpdense_94/MatMul/ReadVariableOp2B
dense_95/BiasAdd/ReadVariableOpdense_95/BiasAdd/ReadVariableOp2@
dense_95/MatMul/ReadVariableOpdense_95/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
í
ß
F__inference_dense_95_layer_call_and_return_conditional_losses_66749788

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
÷
¬
+__inference_dense_95_layer_call_fn_66750050

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_95_layer_call_and_return_conditional_losses_667497882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
È
­
,__inference_conv2d_17_layer_call_fn_66749712

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_17_layer_call_and_return_conditional_losses_667497042
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ù
¬
+__inference_dense_93_layer_call_fn_66750016

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_93_layer_call_and_return_conditional_losses_667497442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
¾


0__inference_sequential_31_layer_call_fn_66749974

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_31_layer_call_and_return_conditional_losses_667498382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Å	
ß
F__inference_dense_93_layer_call_and_return_conditional_losses_66749744

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

d
H__inference_flatten_17_layer_call_and_return_conditional_losses_66749725

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
§


&__inference_signature_wrapper_66749899
conv2d_17_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__wrapped_model_667496922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_17_input
¡2
§
#__inference__wrapped_model_66749692
conv2d_17_input:
6sequential_31_conv2d_17_conv2d_readvariableop_resource;
7sequential_31_conv2d_17_biasadd_readvariableop_resource9
5sequential_31_dense_93_matmul_readvariableop_resource:
6sequential_31_dense_93_biasadd_readvariableop_resource9
5sequential_31_dense_94_matmul_readvariableop_resource:
6sequential_31_dense_94_biasadd_readvariableop_resource9
5sequential_31_dense_95_matmul_readvariableop_resource:
6sequential_31_dense_95_biasadd_readvariableop_resource
identity¢.sequential_31/conv2d_17/BiasAdd/ReadVariableOp¢-sequential_31/conv2d_17/Conv2D/ReadVariableOp¢-sequential_31/dense_93/BiasAdd/ReadVariableOp¢,sequential_31/dense_93/MatMul/ReadVariableOp¢-sequential_31/dense_94/BiasAdd/ReadVariableOp¢,sequential_31/dense_94/MatMul/ReadVariableOp¢-sequential_31/dense_95/BiasAdd/ReadVariableOp¢,sequential_31/dense_95/MatMul/ReadVariableOpÞ
-sequential_31/conv2d_17/Conv2D/ReadVariableOpReadVariableOp6sequential_31_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02/
-sequential_31/conv2d_17/Conv2D/ReadVariableOpö
sequential_31/conv2d_17/Conv2DConv2Dconv2d_17_input5sequential_31/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2 
sequential_31/conv2d_17/Conv2DÕ
.sequential_31/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp7sequential_31_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_31/conv2d_17/BiasAdd/ReadVariableOpé
sequential_31/conv2d_17/BiasAddBiasAdd'sequential_31/conv2d_17/Conv2D:output:06sequential_31/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_31/conv2d_17/BiasAdd
sequential_31/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2 
sequential_31/flatten_17/ConstÕ
 sequential_31/flatten_17/ReshapeReshape(sequential_31/conv2d_17/BiasAdd:output:0'sequential_31/flatten_17/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_31/flatten_17/ReshapeÔ
,sequential_31/dense_93/MatMul/ReadVariableOpReadVariableOp5sequential_31_dense_93_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential_31/dense_93/MatMul/ReadVariableOpÜ
sequential_31/dense_93/MatMulMatMul)sequential_31/flatten_17/Reshape:output:04sequential_31/dense_93/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_31/dense_93/MatMulÒ
-sequential_31/dense_93/BiasAdd/ReadVariableOpReadVariableOp6sequential_31_dense_93_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_31/dense_93/BiasAdd/ReadVariableOpÞ
sequential_31/dense_93/BiasAddBiasAdd'sequential_31/dense_93/MatMul:product:05sequential_31/dense_93/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_31/dense_93/BiasAdd
sequential_31/dense_93/TanhTanh'sequential_31/dense_93/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_31/dense_93/TanhÔ
,sequential_31/dense_94/MatMul/ReadVariableOpReadVariableOp5sequential_31_dense_94_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential_31/dense_94/MatMul/ReadVariableOpÒ
sequential_31/dense_94/MatMulMatMulsequential_31/dense_93/Tanh:y:04sequential_31/dense_94/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_31/dense_94/MatMulÒ
-sequential_31/dense_94/BiasAdd/ReadVariableOpReadVariableOp6sequential_31_dense_94_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_31/dense_94/BiasAdd/ReadVariableOpÞ
sequential_31/dense_94/BiasAddBiasAdd'sequential_31/dense_94/MatMul:product:05sequential_31/dense_94/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_31/dense_94/BiasAddÓ
,sequential_31/dense_95/MatMul/ReadVariableOpReadVariableOp5sequential_31_dense_95_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,sequential_31/dense_95/MatMul/ReadVariableOpÙ
sequential_31/dense_95/MatMulMatMul'sequential_31/dense_94/BiasAdd:output:04sequential_31/dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_31/dense_95/MatMulÑ
-sequential_31/dense_95/BiasAdd/ReadVariableOpReadVariableOp6sequential_31_dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_31/dense_95/BiasAdd/ReadVariableOpÝ
sequential_31/dense_95/BiasAddBiasAdd'sequential_31/dense_95/MatMul:product:05sequential_31/dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_31/dense_95/BiasAddù
IdentityIdentity'sequential_31/dense_95/BiasAdd:output:0/^sequential_31/conv2d_17/BiasAdd/ReadVariableOp.^sequential_31/conv2d_17/Conv2D/ReadVariableOp.^sequential_31/dense_93/BiasAdd/ReadVariableOp-^sequential_31/dense_93/MatMul/ReadVariableOp.^sequential_31/dense_94/BiasAdd/ReadVariableOp-^sequential_31/dense_94/MatMul/ReadVariableOp.^sequential_31/dense_95/BiasAdd/ReadVariableOp-^sequential_31/dense_95/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2`
.sequential_31/conv2d_17/BiasAdd/ReadVariableOp.sequential_31/conv2d_17/BiasAdd/ReadVariableOp2^
-sequential_31/conv2d_17/Conv2D/ReadVariableOp-sequential_31/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_31/dense_93/BiasAdd/ReadVariableOp-sequential_31/dense_93/BiasAdd/ReadVariableOp2\
,sequential_31/dense_93/MatMul/ReadVariableOp,sequential_31/dense_93/MatMul/ReadVariableOp2^
-sequential_31/dense_94/BiasAdd/ReadVariableOp-sequential_31/dense_94/BiasAdd/ReadVariableOp2\
,sequential_31/dense_94/MatMul/ReadVariableOp,sequential_31/dense_94/MatMul/ReadVariableOp2^
-sequential_31/dense_95/BiasAdd/ReadVariableOp-sequential_31/dense_95/BiasAdd/ReadVariableOp2\
,sequential_31/dense_95/MatMul/ReadVariableOp,sequential_31/dense_95/MatMul/ReadVariableOp:/ +
)
_user_specified_nameconv2d_17_input
¿
á
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749868

inputs,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2+
'dense_93_statefulpartitionedcall_args_1+
'dense_93_statefulpartitionedcall_args_2+
'dense_94_statefulpartitionedcall_args_1+
'dense_94_statefulpartitionedcall_args_2+
'dense_95_statefulpartitionedcall_args_1+
'dense_95_statefulpartitionedcall_args_2
identity¢!conv2d_17/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallº
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_17_layer_call_and_return_conditional_losses_667497042#
!conv2d_17/StatefulPartitionedCallë
flatten_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_17_layer_call_and_return_conditional_losses_667497252
flatten_17/PartitionedCallÊ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0'dense_93_statefulpartitionedcall_args_1'dense_93_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_93_layer_call_and_return_conditional_losses_667497442"
 dense_93/StatefulPartitionedCallÐ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0'dense_94_statefulpartitionedcall_args_1'dense_94_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_94_layer_call_and_return_conditional_losses_667497662"
 dense_94/StatefulPartitionedCallÏ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0'dense_95_statefulpartitionedcall_args_1'dense_95_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_95_layer_call_and_return_conditional_losses_667497882"
 dense_95/StatefulPartitionedCall
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
¿
á
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749838

inputs,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2+
'dense_93_statefulpartitionedcall_args_1+
'dense_93_statefulpartitionedcall_args_2+
'dense_94_statefulpartitionedcall_args_1+
'dense_94_statefulpartitionedcall_args_2+
'dense_95_statefulpartitionedcall_args_1+
'dense_95_statefulpartitionedcall_args_2
identity¢!conv2d_17/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallº
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_17_layer_call_and_return_conditional_losses_667497042#
!conv2d_17/StatefulPartitionedCallë
flatten_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_17_layer_call_and_return_conditional_losses_667497252
flatten_17/PartitionedCallÊ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0'dense_93_statefulpartitionedcall_args_1'dense_93_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_93_layer_call_and_return_conditional_losses_667497442"
 dense_93/StatefulPartitionedCallÐ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0'dense_94_statefulpartitionedcall_args_1'dense_94_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_94_layer_call_and_return_conditional_losses_667497662"
 dense_94/StatefulPartitionedCallÏ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0'dense_95_statefulpartitionedcall_args_1'dense_95_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_95_layer_call_and_return_conditional_losses_667497882"
 dense_95/StatefulPartitionedCall
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ú
ê
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749818
conv2d_17_input,
(conv2d_17_statefulpartitionedcall_args_1,
(conv2d_17_statefulpartitionedcall_args_2+
'dense_93_statefulpartitionedcall_args_1+
'dense_93_statefulpartitionedcall_args_2+
'dense_94_statefulpartitionedcall_args_1+
'dense_94_statefulpartitionedcall_args_2+
'dense_95_statefulpartitionedcall_args_1+
'dense_95_statefulpartitionedcall_args_2
identity¢!conv2d_17/StatefulPartitionedCall¢ dense_93/StatefulPartitionedCall¢ dense_94/StatefulPartitionedCall¢ dense_95/StatefulPartitionedCallÃ
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_input(conv2d_17_statefulpartitionedcall_args_1(conv2d_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_17_layer_call_and_return_conditional_losses_667497042#
!conv2d_17/StatefulPartitionedCallë
flatten_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_17_layer_call_and_return_conditional_losses_667497252
flatten_17/PartitionedCallÊ
 dense_93/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0'dense_93_statefulpartitionedcall_args_1'dense_93_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_93_layer_call_and_return_conditional_losses_667497442"
 dense_93/StatefulPartitionedCallÐ
 dense_94/StatefulPartitionedCallStatefulPartitionedCall)dense_93/StatefulPartitionedCall:output:0'dense_94_statefulpartitionedcall_args_1'dense_94_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_94_layer_call_and_return_conditional_losses_667497662"
 dense_94/StatefulPartitionedCallÏ
 dense_95/StatefulPartitionedCallStatefulPartitionedCall)dense_94/StatefulPartitionedCall:output:0'dense_95_statefulpartitionedcall_args_1'dense_95_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_95_layer_call_and_return_conditional_losses_667497882"
 dense_95/StatefulPartitionedCall
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0"^conv2d_17/StatefulPartitionedCall!^dense_93/StatefulPartitionedCall!^dense_94/StatefulPartitionedCall!^dense_95/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_93/StatefulPartitionedCall dense_93/StatefulPartitionedCall2D
 dense_94/StatefulPartitionedCall dense_94/StatefulPartitionedCall2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_17_input
Ù


0__inference_sequential_31_layer_call_fn_66749849
conv2d_17_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_31_layer_call_and_return_conditional_losses_667498382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_17_input

d
H__inference_flatten_17_layer_call_and_return_conditional_losses_66749993

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
í
ß
F__inference_dense_95_layer_call_and_return_conditional_losses_66750043

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
'
ü
!__inference__traced_save_66750116
file_prefix/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop.
*savev2_dense_93_kernel_read_readvariableop,
(savev2_dense_93_bias_read_readvariableop.
*savev2_dense_94_kernel_read_readvariableop,
(savev2_dense_94_bias_read_readvariableop.
*savev2_dense_95_kernel_read_readvariableop,
(savev2_dense_95_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4099a1a5c28f409a88cb907c0d750b25/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameá
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó
valueéBæB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names¤
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesù
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop*savev2_dense_93_kernel_read_readvariableop(savev2_dense_93_bias_read_readvariableop*savev2_dense_94_kernel_read_readvariableop(savev2_dense_94_bias_read_readvariableop*savev2_dense_95_kernel_read_readvariableop(savev2_dense_95_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*t
_input_shapesc
a: :::
::
::	:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
Ë=

$__inference__traced_restore_66750170
file_prefix%
!assignvariableop_conv2d_17_kernel%
!assignvariableop_1_conv2d_17_bias&
"assignvariableop_2_dense_93_kernel$
 assignvariableop_3_dense_93_bias&
"assignvariableop_4_dense_94_kernel$
 assignvariableop_5_dense_94_bias&
"assignvariableop_6_dense_95_kernel$
 assignvariableop_7_dense_95_bias
assignvariableop_8_sgd_iter 
assignvariableop_9_sgd_decay)
%assignvariableop_10_sgd_learning_rate$
 assignvariableop_11_sgd_momentum
assignvariableop_12_total
assignvariableop_13_count
identity_15¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1ç
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó
valueéBæB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesª
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesñ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_17_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_17_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_93_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_93_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_94_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_94_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_95_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_95_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ã
serving_default¯
S
conv2d_17_input@
!serving_default_conv2d_17_input:0ÿÿÿÿÿÿÿÿÿ<
dense_950
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ã±
à+
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
Q_default_save_signature
*R&call_and_return_all_conditional_losses
S__call__"ß(
_tf_keras_sequentialÀ({"class_name": "Sequential", "name": "sequential_31", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_31", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_93", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_94", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_31", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_93", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_94", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [{"class_name": "MeanSquaredError", "config": {"name": "mean_squared_error", "dtype": "float32"}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
¹"¶
_tf_keras_input_layer{"class_name": "InputLayer", "name": "conv2d_17_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 6, 7, 1], "config": {"batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_17_input"}}
¹

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"
_tf_keras_layerú{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 6, 7, 1], "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
²
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"£
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"è
_tf_keras_layerÎ{"class_name": "Dense", "name": "dense_93", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_93", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3072}}}}


kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"é
_tf_keras_layerÏ{"class_name": "Dense", "name": "dense_94", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_94", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
*\&call_and_return_all_conditional_losses
]__call__"ç
_tf_keras_layerÍ{"class_name": "Dense", "name": "dense_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
I
)iter
	*decay
+learning_rate
,momentum"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
·
-metrics
.non_trainable_variables
regularization_losses
		variables

trainable_variables
/layer_regularization_losses

0layers
S__call__
Q_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
,
^serving_default"
signature_map
+:)2conv2d_17/kernel
:2conv2d_17/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

1metrics
2non_trainable_variables
regularization_losses
	variables
trainable_variables
3layer_regularization_losses

4layers
U__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

5metrics
6non_trainable_variables
regularization_losses
	variables
trainable_variables
7layer_regularization_losses

8layers
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_93/kernel
:2dense_93/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

9metrics
:non_trainable_variables
regularization_losses
	variables
trainable_variables
;layer_regularization_losses

<layers
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_94/kernel
:2dense_94/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

=metrics
>non_trainable_variables
regularization_losses
 	variables
!trainable_variables
?layer_regularization_losses

@layers
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_95/kernel
:2dense_95/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper

Ametrics
Bnon_trainable_variables
%regularization_losses
&	variables
'trainable_variables
Clayer_regularization_losses

Dlayers
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
	Ftotal
	Gcount
H
_fn_kwargs
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
*_&call_and_return_all_conditional_losses
`__call__"ø
_tf_keras_layerÞ{"class_name": "MeanSquaredError", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper

Mmetrics
Nnon_trainable_variables
Iregularization_losses
J	variables
Ktrainable_variables
Olayer_regularization_losses

Players
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ñ2î
#__inference__wrapped_model_66749692Æ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *6¢3
1.
conv2d_17_inputÿÿÿÿÿÿÿÿÿ
ú2÷
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749930
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749961
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749818
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749801À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_sequential_31_layer_call_fn_66749879
0__inference_sequential_31_layer_call_fn_66749974
0__inference_sequential_31_layer_call_fn_66749849
0__inference_sequential_31_layer_call_fn_66749987À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
G__inference_conv2d_17_layer_call_and_return_conditional_losses_66749704×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
,__inference_conv2d_17_layer_call_fn_66749712×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ò2ï
H__inference_flatten_17_layer_call_and_return_conditional_losses_66749993¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_flatten_17_layer_call_fn_66749998¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_93_layer_call_and_return_conditional_losses_66750009¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_93_layer_call_fn_66750016¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_94_layer_call_and_return_conditional_losses_66750026¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_94_layer_call_fn_66750033¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_95_layer_call_and_return_conditional_losses_66750043¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_95_layer_call_fn_66750050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
=B;
&__inference_signature_wrapper_66749899conv2d_17_input
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ©
#__inference__wrapped_model_66749692#$@¢=
6¢3
1.
conv2d_17_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_95"
dense_95ÿÿÿÿÿÿÿÿÿÝ
G__inference_conv2d_17_layer_call_and_return_conditional_losses_66749704I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
,__inference_conv2d_17_layer_call_fn_66749712I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
F__inference_dense_93_layer_call_and_return_conditional_losses_66750009^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_93_layer_call_fn_66750016Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_dense_94_layer_call_and_return_conditional_losses_66750026^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_94_layer_call_fn_66750033Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_dense_95_layer_call_and_return_conditional_losses_66750043]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_95_layer_call_fn_66750050P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
H__inference_flatten_17_layer_call_and_return_conditional_losses_66749993b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_flatten_17_layer_call_fn_66749998U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÊ
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749801{#$H¢E
>¢;
1.
conv2d_17_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749818{#$H¢E
>¢;
1.
conv2d_17_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749930r#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_31_layer_call_and_return_conditional_losses_66749961r#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¢
0__inference_sequential_31_layer_call_fn_66749849n#$H¢E
>¢;
1.
conv2d_17_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¢
0__inference_sequential_31_layer_call_fn_66749879n#$H¢E
>¢;
1.
conv2d_17_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_31_layer_call_fn_66749974e#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_31_layer_call_fn_66749987e#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¿
&__inference_signature_wrapper_66749899#$S¢P
¢ 
IªF
D
conv2d_17_input1.
conv2d_17_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_95"
dense_95ÿÿÿÿÿÿÿÿÿ