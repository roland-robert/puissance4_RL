гы
Ў§
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
dtypetypeѕ
Й
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
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.1.02unknown8░И
Ё
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*!
shared_nameconv2d_16/kernel
~
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*'
_output_shapes
:ђ*
dtype0
u
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_16/bias
n
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes	
:ђ*
dtype0
|
dense_90/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_90/kernel
u
#dense_90/kernel/Read/ReadVariableOpReadVariableOpdense_90/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_90/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_90/bias
l
!dense_90/bias/Read/ReadVariableOpReadVariableOpdense_90/bias*
_output_shapes	
:ђ*
dtype0
|
dense_91/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_91/kernel
u
#dense_91/kernel/Read/ReadVariableOpReadVariableOpdense_91/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_91/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_91/bias
l
!dense_91/bias/Read/ReadVariableOpReadVariableOpdense_91/bias*
_output_shapes	
:ђ*
dtype0
{
dense_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_namedense_92/kernel
t
#dense_92/kernel/Read/ReadVariableOpReadVariableOpdense_92/kernel*
_output_shapes
:	ђ*
dtype0
r
dense_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_92/bias
k
!dense_92/bias/Read/ReadVariableOpReadVariableOpdense_92/bias*
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
к
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ђ
valueэBЗ Bь
Д
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
џ
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
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
џ
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
џ
5metrics
6non_trainable_variables
regularization_losses
	variables
trainable_variables
7layer_regularization_losses

8layers
[Y
VARIABLE_VALUEdense_90/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_90/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
џ
9metrics
:non_trainable_variables
regularization_losses
	variables
trainable_variables
;layer_regularization_losses

<layers
[Y
VARIABLE_VALUEdense_91/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_91/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
џ
=metrics
>non_trainable_variables
regularization_losses
 	variables
!trainable_variables
?layer_regularization_losses

@layers
[Y
VARIABLE_VALUEdense_92/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_92/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
џ
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
џ
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
њ
serving_default_conv2d_16_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
«
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_16_inputconv2d_16/kernelconv2d_16/biasdense_90/kerneldense_90/biasdense_91/kerneldense_91/biasdense_92/kerneldense_92/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*/
f*R(
&__inference_signature_wrapper_66749329
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
з
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp#dense_90/kernel/Read/ReadVariableOp!dense_90/bias/Read/ReadVariableOp#dense_91/kernel/Read/ReadVariableOp!dense_91/bias/Read/ReadVariableOp#dense_92/kernel/Read/ReadVariableOp!dense_92/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
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
!__inference__traced_save_66749546
о
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasdense_90/kerneldense_90/biasdense_91/kerneldense_91/biasdense_92/kerneldense_92/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
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
$__inference__traced_restore_66749600из
┐
р
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749268

inputs,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2+
'dense_90_statefulpartitionedcall_args_1+
'dense_90_statefulpartitionedcall_args_2+
'dense_91_statefulpartitionedcall_args_1+
'dense_91_statefulpartitionedcall_args_2+
'dense_92_statefulpartitionedcall_args_1+
'dense_92_statefulpartitionedcall_args_2
identityѕб!conv2d_16/StatefulPartitionedCallб dense_90/StatefulPartitionedCallб dense_91/StatefulPartitionedCallб dense_92/StatefulPartitionedCall║
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_16_layer_call_and_return_conditional_losses_667491342#
!conv2d_16/StatefulPartitionedCallв
flatten_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_667491552
flatten_16/PartitionedCall╩
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0'dense_90_statefulpartitionedcall_args_1'dense_90_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_90_layer_call_and_return_conditional_losses_667491742"
 dense_90/StatefulPartitionedCallл
 dense_91/StatefulPartitionedCallStatefulPartitionedCall)dense_90/StatefulPartitionedCall:output:0'dense_91_statefulpartitionedcall_args_1'dense_91_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_91_layer_call_and_return_conditional_losses_667491962"
 dense_91/StatefulPartitionedCall¤
 dense_92/StatefulPartitionedCallStatefulPartitionedCall)dense_91/StatefulPartitionedCall:output:0'dense_92_statefulpartitionedcall_args_1'dense_92_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_92_layer_call_and_return_conditional_losses_667492182"
 dense_92/StatefulPartitionedCallі
IdentityIdentity)dense_92/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall!^dense_90/StatefulPartitionedCall!^dense_91/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2D
 dense_91/StatefulPartitionedCall dense_91/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Д

ѕ
&__inference_signature_wrapper_66749329
conv2d_16_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__wrapped_model_667491222
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_16_input
љ
d
H__inference_flatten_16_layer_call_and_return_conditional_losses_66749155

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
Й

Ѕ
0__inference_sequential_30_layer_call_fn_66749404

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_30_layer_call_and_return_conditional_losses_667492682
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
А2
Д
#__inference__wrapped_model_66749122
conv2d_16_input:
6sequential_30_conv2d_16_conv2d_readvariableop_resource;
7sequential_30_conv2d_16_biasadd_readvariableop_resource9
5sequential_30_dense_90_matmul_readvariableop_resource:
6sequential_30_dense_90_biasadd_readvariableop_resource9
5sequential_30_dense_91_matmul_readvariableop_resource:
6sequential_30_dense_91_biasadd_readvariableop_resource9
5sequential_30_dense_92_matmul_readvariableop_resource:
6sequential_30_dense_92_biasadd_readvariableop_resource
identityѕб.sequential_30/conv2d_16/BiasAdd/ReadVariableOpб-sequential_30/conv2d_16/Conv2D/ReadVariableOpб-sequential_30/dense_90/BiasAdd/ReadVariableOpб,sequential_30/dense_90/MatMul/ReadVariableOpб-sequential_30/dense_91/BiasAdd/ReadVariableOpб,sequential_30/dense_91/MatMul/ReadVariableOpб-sequential_30/dense_92/BiasAdd/ReadVariableOpб,sequential_30/dense_92/MatMul/ReadVariableOpя
-sequential_30/conv2d_16/Conv2D/ReadVariableOpReadVariableOp6sequential_30_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype02/
-sequential_30/conv2d_16/Conv2D/ReadVariableOpШ
sequential_30/conv2d_16/Conv2DConv2Dconv2d_16_input5sequential_30/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2 
sequential_30/conv2d_16/Conv2DН
.sequential_30/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp7sequential_30_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype020
.sequential_30/conv2d_16/BiasAdd/ReadVariableOpж
sequential_30/conv2d_16/BiasAddBiasAdd'sequential_30/conv2d_16/Conv2D:output:06sequential_30/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2!
sequential_30/conv2d_16/BiasAddЉ
sequential_30/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
sequential_30/flatten_16/ConstН
 sequential_30/flatten_16/ReshapeReshape(sequential_30/conv2d_16/BiasAdd:output:0'sequential_30/flatten_16/Const:output:0*
T0*(
_output_shapes
:         ђ2"
 sequential_30/flatten_16/Reshapeн
,sequential_30/dense_90/MatMul/ReadVariableOpReadVariableOp5sequential_30_dense_90_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02.
,sequential_30/dense_90/MatMul/ReadVariableOp▄
sequential_30/dense_90/MatMulMatMul)sequential_30/flatten_16/Reshape:output:04sequential_30/dense_90/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_30/dense_90/MatMulм
-sequential_30/dense_90/BiasAdd/ReadVariableOpReadVariableOp6sequential_30_dense_90_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_30/dense_90/BiasAdd/ReadVariableOpя
sequential_30/dense_90/BiasAddBiasAdd'sequential_30/dense_90/MatMul:product:05sequential_30/dense_90/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2 
sequential_30/dense_90/BiasAddъ
sequential_30/dense_90/TanhTanh'sequential_30/dense_90/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_30/dense_90/Tanhн
,sequential_30/dense_91/MatMul/ReadVariableOpReadVariableOp5sequential_30_dense_91_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02.
,sequential_30/dense_91/MatMul/ReadVariableOpм
sequential_30/dense_91/MatMulMatMulsequential_30/dense_90/Tanh:y:04sequential_30/dense_91/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_30/dense_91/MatMulм
-sequential_30/dense_91/BiasAdd/ReadVariableOpReadVariableOp6sequential_30_dense_91_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_30/dense_91/BiasAdd/ReadVariableOpя
sequential_30/dense_91/BiasAddBiasAdd'sequential_30/dense_91/MatMul:product:05sequential_30/dense_91/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2 
sequential_30/dense_91/BiasAddМ
,sequential_30/dense_92/MatMul/ReadVariableOpReadVariableOp5sequential_30_dense_92_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02.
,sequential_30/dense_92/MatMul/ReadVariableOp┘
sequential_30/dense_92/MatMulMatMul'sequential_30/dense_91/BiasAdd:output:04sequential_30/dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_30/dense_92/MatMulЛ
-sequential_30/dense_92/BiasAdd/ReadVariableOpReadVariableOp6sequential_30_dense_92_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_30/dense_92/BiasAdd/ReadVariableOpП
sequential_30/dense_92/BiasAddBiasAdd'sequential_30/dense_92/MatMul:product:05sequential_30/dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential_30/dense_92/BiasAddщ
IdentityIdentity'sequential_30/dense_92/BiasAdd:output:0/^sequential_30/conv2d_16/BiasAdd/ReadVariableOp.^sequential_30/conv2d_16/Conv2D/ReadVariableOp.^sequential_30/dense_90/BiasAdd/ReadVariableOp-^sequential_30/dense_90/MatMul/ReadVariableOp.^sequential_30/dense_91/BiasAdd/ReadVariableOp-^sequential_30/dense_91/MatMul/ReadVariableOp.^sequential_30/dense_92/BiasAdd/ReadVariableOp-^sequential_30/dense_92/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::2`
.sequential_30/conv2d_16/BiasAdd/ReadVariableOp.sequential_30/conv2d_16/BiasAdd/ReadVariableOp2^
-sequential_30/conv2d_16/Conv2D/ReadVariableOp-sequential_30/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_30/dense_90/BiasAdd/ReadVariableOp-sequential_30/dense_90/BiasAdd/ReadVariableOp2\
,sequential_30/dense_90/MatMul/ReadVariableOp,sequential_30/dense_90/MatMul/ReadVariableOp2^
-sequential_30/dense_91/BiasAdd/ReadVariableOp-sequential_30/dense_91/BiasAdd/ReadVariableOp2\
,sequential_30/dense_91/MatMul/ReadVariableOp,sequential_30/dense_91/MatMul/ReadVariableOp2^
-sequential_30/dense_92/BiasAdd/ReadVariableOp-sequential_30/dense_92/BiasAdd/ReadVariableOp2\
,sequential_30/dense_92/MatMul/ReadVariableOp,sequential_30/dense_92/MatMul/ReadVariableOp:/ +
)
_user_specified_nameconv2d_16_input
э
г
+__inference_dense_92_layer_call_fn_66749480

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_92_layer_call_and_return_conditional_losses_667492182
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ы
▀
F__inference_dense_91_layer_call_and_return_conditional_losses_66749456

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Й

Ѕ
0__inference_sequential_30_layer_call_fn_66749417

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_30_layer_call_and_return_conditional_losses_667492982
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ч

Я
G__inference_conv2d_16_layer_call_and_return_conditional_losses_66749134

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype02
Conv2D/ReadVariableOpи
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЏ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ2	
BiasAdd░
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
┌
Ж
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749248
conv2d_16_input,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2+
'dense_90_statefulpartitionedcall_args_1+
'dense_90_statefulpartitionedcall_args_2+
'dense_91_statefulpartitionedcall_args_1+
'dense_91_statefulpartitionedcall_args_2+
'dense_92_statefulpartitionedcall_args_1+
'dense_92_statefulpartitionedcall_args_2
identityѕб!conv2d_16/StatefulPartitionedCallб dense_90/StatefulPartitionedCallб dense_91/StatefulPartitionedCallб dense_92/StatefulPartitionedCall├
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_input(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_16_layer_call_and_return_conditional_losses_667491342#
!conv2d_16/StatefulPartitionedCallв
flatten_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_667491552
flatten_16/PartitionedCall╩
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0'dense_90_statefulpartitionedcall_args_1'dense_90_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_90_layer_call_and_return_conditional_losses_667491742"
 dense_90/StatefulPartitionedCallл
 dense_91/StatefulPartitionedCallStatefulPartitionedCall)dense_90/StatefulPartitionedCall:output:0'dense_91_statefulpartitionedcall_args_1'dense_91_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_91_layer_call_and_return_conditional_losses_667491962"
 dense_91/StatefulPartitionedCall¤
 dense_92/StatefulPartitionedCallStatefulPartitionedCall)dense_91/StatefulPartitionedCall:output:0'dense_92_statefulpartitionedcall_args_1'dense_92_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_92_layer_call_and_return_conditional_losses_667492182"
 dense_92/StatefulPartitionedCallі
IdentityIdentity)dense_92/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall!^dense_90/StatefulPartitionedCall!^dense_91/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2D
 dense_91/StatefulPartitionedCall dense_91/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_16_input
┘

њ
0__inference_sequential_30_layer_call_fn_66749309
conv2d_16_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_30_layer_call_and_return_conditional_losses_667492982
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_16_input
┌
Ж
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749231
conv2d_16_input,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2+
'dense_90_statefulpartitionedcall_args_1+
'dense_90_statefulpartitionedcall_args_2+
'dense_91_statefulpartitionedcall_args_1+
'dense_91_statefulpartitionedcall_args_2+
'dense_92_statefulpartitionedcall_args_1+
'dense_92_statefulpartitionedcall_args_2
identityѕб!conv2d_16/StatefulPartitionedCallб dense_90/StatefulPartitionedCallб dense_91/StatefulPartitionedCallб dense_92/StatefulPartitionedCall├
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_input(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_16_layer_call_and_return_conditional_losses_667491342#
!conv2d_16/StatefulPartitionedCallв
flatten_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_667491552
flatten_16/PartitionedCall╩
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0'dense_90_statefulpartitionedcall_args_1'dense_90_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_90_layer_call_and_return_conditional_losses_667491742"
 dense_90/StatefulPartitionedCallл
 dense_91/StatefulPartitionedCallStatefulPartitionedCall)dense_90/StatefulPartitionedCall:output:0'dense_91_statefulpartitionedcall_args_1'dense_91_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_91_layer_call_and_return_conditional_losses_667491962"
 dense_91/StatefulPartitionedCall¤
 dense_92/StatefulPartitionedCallStatefulPartitionedCall)dense_91/StatefulPartitionedCall:output:0'dense_92_statefulpartitionedcall_args_1'dense_92_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_92_layer_call_and_return_conditional_losses_667492182"
 dense_92/StatefulPartitionedCallі
IdentityIdentity)dense_92/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall!^dense_90/StatefulPartitionedCall!^dense_91/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2D
 dense_91/StatefulPartitionedCall dense_91/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_16_input
џ'
Ч
!__inference__traced_save_66749546
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop.
*savev2_dense_90_kernel_read_readvariableop,
(savev2_dense_90_bias_read_readvariableop.
*savev2_dense_91_kernel_read_readvariableop,
(savev2_dense_91_bias_read_readvariableop.
*savev2_dense_92_kernel_read_readvariableop,
(savev2_dense_92_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ц
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_fadaa82ab54349599a5a1219bd3f0b81/part2
StringJoin/inputs_1Ђ

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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameр
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*з
valueжBТB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesц
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesщ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop*savev2_dense_90_kernel_read_readvariableop(savev2_dense_90_bias_read_readvariableop*savev2_dense_91_kernel_read_readvariableop(savev2_dense_91_bias_read_readvariableop*savev2_dense_92_kernel_read_readvariableop(savev2_dense_92_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*t
_input_shapesc
a: :ђ:ђ:
ђђ:ђ:
ђђ:ђ:	ђ:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
┘

њ
0__inference_sequential_30_layer_call_fn_66749279
conv2d_16_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_30_layer_call_and_return_conditional_losses_667492682
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_16_input
Ь&
Т
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749360

inputs,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource+
'dense_90_matmul_readvariableop_resource,
(dense_90_biasadd_readvariableop_resource+
'dense_91_matmul_readvariableop_resource,
(dense_91_biasadd_readvariableop_resource+
'dense_92_matmul_readvariableop_resource,
(dense_92_biasadd_readvariableop_resource
identityѕб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpбdense_90/BiasAdd/ReadVariableOpбdense_90/MatMul/ReadVariableOpбdense_91/BiasAdd/ReadVariableOpбdense_91/MatMul/ReadVariableOpбdense_92/BiasAdd/ReadVariableOpбdense_92/MatMul/ReadVariableOp┤
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype02!
conv2d_16/Conv2D/ReadVariableOp├
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2
conv2d_16/Conv2DФ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp▒
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_16/BiasAddu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_16/ConstЮ
flatten_16/ReshapeReshapeconv2d_16/BiasAdd:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_16/Reshapeф
dense_90/MatMul/ReadVariableOpReadVariableOp'dense_90_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_90/MatMul/ReadVariableOpц
dense_90/MatMulMatMulflatten_16/Reshape:output:0&dense_90/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_90/MatMulе
dense_90/BiasAdd/ReadVariableOpReadVariableOp(dense_90_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_90/BiasAdd/ReadVariableOpд
dense_90/BiasAddBiasAdddense_90/MatMul:product:0'dense_90/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_90/BiasAddt
dense_90/TanhTanhdense_90/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_90/Tanhф
dense_91/MatMul/ReadVariableOpReadVariableOp'dense_91_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_91/MatMul/ReadVariableOpџ
dense_91/MatMulMatMuldense_90/Tanh:y:0&dense_91/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_91/MatMulе
dense_91/BiasAdd/ReadVariableOpReadVariableOp(dense_91_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_91/BiasAdd/ReadVariableOpд
dense_91/BiasAddBiasAdddense_91/MatMul:product:0'dense_91/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_91/BiasAddЕ
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_92/MatMul/ReadVariableOpА
dense_92/MatMulMatMuldense_91/BiasAdd:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_92/MatMulД
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_92/BiasAdd/ReadVariableOpЦ
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_92/BiasAddч
IdentityIdentitydense_92/BiasAdd:output:0!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp ^dense_90/BiasAdd/ReadVariableOp^dense_90/MatMul/ReadVariableOp ^dense_91/BiasAdd/ReadVariableOp^dense_91/MatMul/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2B
dense_90/BiasAdd/ReadVariableOpdense_90/BiasAdd/ReadVariableOp2@
dense_90/MatMul/ReadVariableOpdense_90/MatMul/ReadVariableOp2B
dense_91/BiasAdd/ReadVariableOpdense_91/BiasAdd/ReadVariableOp2@
dense_91/MatMul/ReadVariableOpdense_91/MatMul/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
┐
р
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749298

inputs,
(conv2d_16_statefulpartitionedcall_args_1,
(conv2d_16_statefulpartitionedcall_args_2+
'dense_90_statefulpartitionedcall_args_1+
'dense_90_statefulpartitionedcall_args_2+
'dense_91_statefulpartitionedcall_args_1+
'dense_91_statefulpartitionedcall_args_2+
'dense_92_statefulpartitionedcall_args_1+
'dense_92_statefulpartitionedcall_args_2
identityѕб!conv2d_16/StatefulPartitionedCallб dense_90/StatefulPartitionedCallб dense_91/StatefulPartitionedCallб dense_92/StatefulPartitionedCall║
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_16_statefulpartitionedcall_args_1(conv2d_16_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_16_layer_call_and_return_conditional_losses_667491342#
!conv2d_16/StatefulPartitionedCallв
flatten_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_667491552
flatten_16/PartitionedCall╩
 dense_90/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0'dense_90_statefulpartitionedcall_args_1'dense_90_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_90_layer_call_and_return_conditional_losses_667491742"
 dense_90/StatefulPartitionedCallл
 dense_91/StatefulPartitionedCallStatefulPartitionedCall)dense_90/StatefulPartitionedCall:output:0'dense_91_statefulpartitionedcall_args_1'dense_91_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_91_layer_call_and_return_conditional_losses_667491962"
 dense_91/StatefulPartitionedCall¤
 dense_92/StatefulPartitionedCallStatefulPartitionedCall)dense_91/StatefulPartitionedCall:output:0'dense_92_statefulpartitionedcall_args_1'dense_92_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_92_layer_call_and_return_conditional_losses_667492182"
 dense_92/StatefulPartitionedCallі
IdentityIdentity)dense_92/StatefulPartitionedCall:output:0"^conv2d_16/StatefulPartitionedCall!^dense_90/StatefulPartitionedCall!^dense_91/StatefulPartitionedCall!^dense_92/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2D
 dense_90/StatefulPartitionedCall dense_90/StatefulPartitionedCall2D
 dense_91/StatefulPartitionedCall dense_91/StatefulPartitionedCall2D
 dense_92/StatefulPartitionedCall dense_92/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
┼	
▀
F__inference_dense_90_layer_call_and_return_conditional_losses_66749439

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Tanhј
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╚
Г
,__inference_conv2d_16_layer_call_fn_66749142

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           ђ**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_16_layer_call_and_return_conditional_losses_667491342
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
щ
г
+__inference_dense_91_layer_call_fn_66749463

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_91_layer_call_and_return_conditional_losses_667491962
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┼	
▀
F__inference_dense_90_layer_call_and_return_conditional_losses_66749174

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Tanhј
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ы
▀
F__inference_dense_91_layer_call_and_return_conditional_losses_66749196

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╦=
Ќ
$__inference__traced_restore_66749600
file_prefix%
!assignvariableop_conv2d_16_kernel%
!assignvariableop_1_conv2d_16_bias&
"assignvariableop_2_dense_90_kernel$
 assignvariableop_3_dense_90_bias&
"assignvariableop_4_dense_91_kernel$
 assignvariableop_5_dense_91_bias&
"assignvariableop_6_dense_92_kernel$
 assignvariableop_7_dense_92_bias
assignvariableop_8_sgd_iter 
assignvariableop_9_sgd_decay)
%assignvariableop_10_sgd_learning_rate$
 assignvariableop_11_sgd_momentum
assignvariableop_12_total
assignvariableop_13_count
identity_15ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1у
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*з
valueжBТB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesф
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesы
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

IdentityЉ
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ќ
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2ў
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_90_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3ќ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_90_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4ў
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_91_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5ќ
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_91_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6ў
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_92_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7ќ
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_92_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8Љ
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9њ
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10ъ
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ў
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12њ
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13њ
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
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
NoOpњ
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14Ъ
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
_user_specified_namefile_prefix
ь
▀
F__inference_dense_92_layer_call_and_return_conditional_losses_66749218

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
щ
г
+__inference_dense_90_layer_call_fn_66749446

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_90_layer_call_and_return_conditional_losses_667491742
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ь
▀
F__inference_dense_92_layer_call_and_return_conditional_losses_66749473

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ь&
Т
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749391

inputs,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource+
'dense_90_matmul_readvariableop_resource,
(dense_90_biasadd_readvariableop_resource+
'dense_91_matmul_readvariableop_resource,
(dense_91_biasadd_readvariableop_resource+
'dense_92_matmul_readvariableop_resource,
(dense_92_biasadd_readvariableop_resource
identityѕб conv2d_16/BiasAdd/ReadVariableOpбconv2d_16/Conv2D/ReadVariableOpбdense_90/BiasAdd/ReadVariableOpбdense_90/MatMul/ReadVariableOpбdense_91/BiasAdd/ReadVariableOpбdense_91/MatMul/ReadVariableOpбdense_92/BiasAdd/ReadVariableOpбdense_92/MatMul/ReadVariableOp┤
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype02!
conv2d_16/Conv2D/ReadVariableOp├
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2
conv2d_16/Conv2DФ
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp▒
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_16/BiasAddu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_16/ConstЮ
flatten_16/ReshapeReshapeconv2d_16/BiasAdd:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_16/Reshapeф
dense_90/MatMul/ReadVariableOpReadVariableOp'dense_90_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_90/MatMul/ReadVariableOpц
dense_90/MatMulMatMulflatten_16/Reshape:output:0&dense_90/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_90/MatMulе
dense_90/BiasAdd/ReadVariableOpReadVariableOp(dense_90_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_90/BiasAdd/ReadVariableOpд
dense_90/BiasAddBiasAdddense_90/MatMul:product:0'dense_90/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_90/BiasAddt
dense_90/TanhTanhdense_90/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_90/Tanhф
dense_91/MatMul/ReadVariableOpReadVariableOp'dense_91_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_91/MatMul/ReadVariableOpџ
dense_91/MatMulMatMuldense_90/Tanh:y:0&dense_91/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_91/MatMulе
dense_91/BiasAdd/ReadVariableOpReadVariableOp(dense_91_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_91/BiasAdd/ReadVariableOpд
dense_91/BiasAddBiasAdddense_91/MatMul:product:0'dense_91/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_91/BiasAddЕ
dense_92/MatMul/ReadVariableOpReadVariableOp'dense_92_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_92/MatMul/ReadVariableOpА
dense_92/MatMulMatMuldense_91/BiasAdd:output:0&dense_92/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_92/MatMulД
dense_92/BiasAdd/ReadVariableOpReadVariableOp(dense_92_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_92/BiasAdd/ReadVariableOpЦ
dense_92/BiasAddBiasAdddense_92/MatMul:product:0'dense_92/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_92/BiasAddч
IdentityIdentitydense_92/BiasAdd:output:0!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp ^dense_90/BiasAdd/ReadVariableOp^dense_90/MatMul/ReadVariableOp ^dense_91/BiasAdd/ReadVariableOp^dense_91/MatMul/ReadVariableOp ^dense_92/BiasAdd/ReadVariableOp^dense_92/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2B
dense_90/BiasAdd/ReadVariableOpdense_90/BiasAdd/ReadVariableOp2@
dense_90/MatMul/ReadVariableOpdense_90/MatMul/ReadVariableOp2B
dense_91/BiasAdd/ReadVariableOpdense_91/BiasAdd/ReadVariableOp2@
dense_91/MatMul/ReadVariableOpdense_91/MatMul/ReadVariableOp2B
dense_92/BiasAdd/ReadVariableOpdense_92/BiasAdd/ReadVariableOp2@
dense_92/MatMul/ReadVariableOpdense_92/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
љ
d
H__inference_flatten_16_layer_call_and_return_conditional_losses_66749423

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
Т
I
-__inference_flatten_16_layer_call_fn_66749428

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ђ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_667491552
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs"»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*├
serving_default»
S
conv2d_16_input@
!serving_default_conv2d_16_input:0         <
dense_920
StatefulPartitionedCall:0         tensorflow/serving/predict:с▒
Я+
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
S__call__"▀(
_tf_keras_sequential└({"class_name": "Sequential", "name": "sequential_30", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_30", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_90", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_91", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_92", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_30", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_90", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_91", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_92", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [{"class_name": "MeanSquaredError", "config": {"name": "mean_squared_error", "dtype": "float32"}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
╣"Х
_tf_keras_input_layerќ{"class_name": "InputLayer", "name": "conv2d_16_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 6, 7, 1], "config": {"batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_16_input"}}
╣

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"ћ
_tf_keras_layerЩ{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 6, 7, 1], "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
▓
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"Б
_tf_keras_layerЅ{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ї

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"У
_tf_keras_layer╬{"class_name": "Dense", "name": "dense_90", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_90", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3072}}}}
ј

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"ж
_tf_keras_layer¤{"class_name": "Dense", "name": "dense_91", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_91", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
ї

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
*\&call_and_return_all_conditional_losses
]__call__"у
_tf_keras_layer═{"class_name": "Dense", "name": "dense_92", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_92", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
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
и
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
+:)ђ2conv2d_16/kernel
:ђ2conv2d_16/bias
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
џ
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
џ
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
ђђ2dense_90/kernel
:ђ2dense_90/bias
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
џ
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
ђђ2dense_91/kernel
:ђ2dense_91/bias
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
џ
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
": 	ђ2dense_92/kernel
:2dense_92/bias
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
џ
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
Г
	Ftotal
	Gcount
H
_fn_kwargs
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
*_&call_and_return_all_conditional_losses
`__call__"Э
_tf_keras_layerя{"class_name": "MeanSquaredError", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
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
џ
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
ы2Ь
#__inference__wrapped_model_66749122к
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *6б3
1і.
conv2d_16_input         
Щ2э
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749391
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749231
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749360
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749248└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
0__inference_sequential_30_layer_call_fn_66749404
0__inference_sequential_30_layer_call_fn_66749309
0__inference_sequential_30_layer_call_fn_66749417
0__inference_sequential_30_layer_call_fn_66749279└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
д2Б
G__inference_conv2d_16_layer_call_and_return_conditional_losses_66749134О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
І2ѕ
,__inference_conv2d_16_layer_call_fn_66749142О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
Ы2№
H__inference_flatten_16_layer_call_and_return_conditional_losses_66749423б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_flatten_16_layer_call_fn_66749428б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_90_layer_call_and_return_conditional_losses_66749439б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_90_layer_call_fn_66749446б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_91_layer_call_and_return_conditional_losses_66749456б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_91_layer_call_fn_66749463б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_92_layer_call_and_return_conditional_losses_66749473б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_92_layer_call_fn_66749480б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
=B;
&__inference_signature_wrapper_66749329conv2d_16_input
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 Е
#__inference__wrapped_model_66749122Ђ#$@б=
6б3
1і.
conv2d_16_input         
ф "3ф0
.
dense_92"і
dense_92         П
G__inference_conv2d_16_layer_call_and_return_conditional_losses_66749134ЉIбF
?б<
:і7
inputs+                           
ф "@б=
6і3
0,                           ђ
џ х
,__inference_conv2d_16_layer_call_fn_66749142ёIбF
?б<
:і7
inputs+                           
ф "3і0,                           ђе
F__inference_dense_90_layer_call_and_return_conditional_losses_66749439^0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ђ
+__inference_dense_90_layer_call_fn_66749446Q0б-
&б#
!і
inputs         ђ
ф "і         ђе
F__inference_dense_91_layer_call_and_return_conditional_losses_66749456^0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ђ
+__inference_dense_91_layer_call_fn_66749463Q0б-
&б#
!і
inputs         ђ
ф "і         ђД
F__inference_dense_92_layer_call_and_return_conditional_losses_66749473]#$0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ 
+__inference_dense_92_layer_call_fn_66749480P#$0б-
&б#
!і
inputs         ђ
ф "і         «
H__inference_flatten_16_layer_call_and_return_conditional_losses_66749423b8б5
.б+
)і&
inputs         ђ
ф "&б#
і
0         ђ
џ є
-__inference_flatten_16_layer_call_fn_66749428U8б5
.б+
)і&
inputs         ђ
ф "і         ђ╩
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749231{#$HбE
>б;
1і.
conv2d_16_input         
p

 
ф "%б"
і
0         
џ ╩
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749248{#$HбE
>б;
1і.
conv2d_16_input         
p 

 
ф "%б"
і
0         
џ ┴
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749360r#$?б<
5б2
(і%
inputs         
p

 
ф "%б"
і
0         
џ ┴
K__inference_sequential_30_layer_call_and_return_conditional_losses_66749391r#$?б<
5б2
(і%
inputs         
p 

 
ф "%б"
і
0         
џ б
0__inference_sequential_30_layer_call_fn_66749279n#$HбE
>б;
1і.
conv2d_16_input         
p

 
ф "і         б
0__inference_sequential_30_layer_call_fn_66749309n#$HбE
>б;
1і.
conv2d_16_input         
p 

 
ф "і         Ў
0__inference_sequential_30_layer_call_fn_66749404e#$?б<
5б2
(і%
inputs         
p

 
ф "і         Ў
0__inference_sequential_30_layer_call_fn_66749417e#$?б<
5б2
(і%
inputs         
p 

 
ф "і         ┐
&__inference_signature_wrapper_66749329ћ#$SбP
б 
IфF
D
conv2d_16_input1і.
conv2d_16_input         "3ф0
.
dense_92"і
dense_92         