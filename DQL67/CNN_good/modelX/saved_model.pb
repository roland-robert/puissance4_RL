úõ
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
shapeshape"serve*2.1.02unknown8²¼

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_20/kernel
~
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*'
_output_shapes
:*
dtype0
u
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
n
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes	
:*
dtype0
~
dense_102/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
<*!
shared_namedense_102/kernel
w
$dense_102/kernel/Read/ReadVariableOpReadVariableOpdense_102/kernel* 
_output_shapes
:
<*
dtype0
u
dense_102/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_102/bias
n
"dense_102/bias/Read/ReadVariableOpReadVariableOpdense_102/bias*
_output_shapes	
:*
dtype0
~
dense_103/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_103/kernel
w
$dense_103/kernel/Read/ReadVariableOpReadVariableOpdense_103/kernel* 
_output_shapes
:
*
dtype0
u
dense_103/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_103/bias
n
"dense_103/bias/Read/ReadVariableOpReadVariableOpdense_103/bias*
_output_shapes	
:*
dtype0
}
dense_104/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_104/kernel
v
$dense_104/kernel/Read/ReadVariableOpReadVariableOpdense_104/kernel*
_output_shapes
:	*
dtype0
t
dense_104/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_104/bias
m
"dense_104/bias/Read/ReadVariableOpReadVariableOpdense_104/bias*
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
Ì
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueýBú Bó
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
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_102/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_102/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_103/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_103/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_104/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_104/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_conv2d_20_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
µ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_20_inputconv2d_20/kernelconv2d_20/biasdense_102/kerneldense_102/biasdense_103/kerneldense_103/biasdense_104/kerneldense_104/bias*
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
GPU 2J 8*0
f+R)
'__inference_signature_wrapper_180406708
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$dense_102/kernel/Read/ReadVariableOp"dense_102/bias/Read/ReadVariableOp$dense_103/kernel/Read/ReadVariableOp"dense_103/bias/Read/ReadVariableOp$dense_104/kernel/Read/ReadVariableOp"dense_104/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
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
GPU 2J 8*+
f&R$
"__inference__traced_save_180406925
Ý
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_20/kernelconv2d_20/biasdense_102/kerneldense_102/biasdense_103/kerneldense_103/biasdense_104/kerneldense_104/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
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
GPU 2J 8*.
f)R'
%__inference__traced_restore_180406979÷
©


'__inference_signature_wrapper_180406708
conv2d_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
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
GPU 2J 8*-
f(R&
$__inference__wrapped_model_1804065012
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
_user_specified_nameconv2d_20_input
ý
®
-__inference_dense_102_layer_call_fn_180406825

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
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
GPU 2J 8*Q
fLRJ
H__inference_dense_102_layer_call_and_return_conditional_losses_1804065532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ô
á
H__inference_dense_103_layer_call_and_return_conditional_losses_180406835

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
û
®
-__inference_dense_104_layer_call_fn_180406859

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
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
GPU 2J 8*Q
fLRJ
H__inference_dense_104_layer_call_and_return_conditional_losses_1804065972
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
ô
á
H__inference_dense_103_layer_call_and_return_conditional_losses_180406575

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
é
ë
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406677

inputs,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2,
(dense_103_statefulpartitionedcall_args_1,
(dense_103_statefulpartitionedcall_args_2,
(dense_104_statefulpartitionedcall_args_1,
(dense_104_statefulpartitionedcall_args_2
identity¢!conv2d_20/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCall»
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_20_layer_call_and_return_conditional_losses_1804065132#
!conv2d_20/StatefulPartitionedCallì
flatten_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_20_layer_call_and_return_conditional_losses_1804065342
flatten_20/PartitionedCallÐ
!dense_102/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_102_layer_call_and_return_conditional_losses_1804065532#
!dense_102/StatefulPartitionedCall×
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0(dense_103_statefulpartitionedcall_args_1(dense_103_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_103_layer_call_and_return_conditional_losses_1804065752#
!dense_103/StatefulPartitionedCallÖ
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0(dense_104_statefulpartitionedcall_args_1(dense_104_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_104_layer_call_and_return_conditional_losses_1804065972#
!dense_104/StatefulPartitionedCall
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^conv2d_20/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
À


1__inference_sequential_34_layer_call_fn_180406783

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÔ
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
GPU 2J 8*U
fPRN
L__inference_sequential_34_layer_call_and_return_conditional_losses_1804066472
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
Ç	
á
H__inference_dense_102_layer_call_and_return_conditional_losses_180406553

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
<*
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
:ÿÿÿÿÿÿÿÿÿ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

e
I__inference_flatten_20_layer_call_and_return_conditional_losses_180406534

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs

ô
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406610
conv2d_20_input,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2,
(dense_103_statefulpartitionedcall_args_1,
(dense_103_statefulpartitionedcall_args_2,
(dense_104_statefulpartitionedcall_args_1,
(dense_104_statefulpartitionedcall_args_2
identity¢!conv2d_20/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCallÄ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallconv2d_20_input(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_20_layer_call_and_return_conditional_losses_1804065132#
!conv2d_20/StatefulPartitionedCallì
flatten_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_20_layer_call_and_return_conditional_losses_1804065342
flatten_20/PartitionedCallÐ
!dense_102/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_102_layer_call_and_return_conditional_losses_1804065532#
!dense_102/StatefulPartitionedCall×
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0(dense_103_statefulpartitionedcall_args_1(dense_103_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_103_layer_call_and_return_conditional_losses_1804065752#
!dense_103/StatefulPartitionedCallÖ
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0(dense_104_statefulpartitionedcall_args_1(dense_104_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_104_layer_call_and_return_conditional_losses_1804065972#
!dense_104/StatefulPartitionedCall
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^conv2d_20/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_20_input
Ê
®
-__inference_conv2d_20_layer_call_fn_180406521

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall¥
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
GPU 2J 8*Q
fLRJ
H__inference_conv2d_20_layer_call_and_return_conditional_losses_1804065132
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
ï
á
H__inference_dense_104_layer_call_and_return_conditional_losses_180406852

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
Û


1__inference_sequential_34_layer_call_fn_180406658
conv2d_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
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
GPU 2J 8*U
fPRN
L__inference_sequential_34_layer_call_and_return_conditional_losses_1804066472
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
_user_specified_nameconv2d_20_input
Ç	
á
H__inference_dense_102_layer_call_and_return_conditional_losses_180406818

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
<*
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
:ÿÿÿÿÿÿÿÿÿ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
í2
´
$__inference__wrapped_model_180406501
conv2d_20_input:
6sequential_34_conv2d_20_conv2d_readvariableop_resource;
7sequential_34_conv2d_20_biasadd_readvariableop_resource:
6sequential_34_dense_102_matmul_readvariableop_resource;
7sequential_34_dense_102_biasadd_readvariableop_resource:
6sequential_34_dense_103_matmul_readvariableop_resource;
7sequential_34_dense_103_biasadd_readvariableop_resource:
6sequential_34_dense_104_matmul_readvariableop_resource;
7sequential_34_dense_104_biasadd_readvariableop_resource
identity¢.sequential_34/conv2d_20/BiasAdd/ReadVariableOp¢-sequential_34/conv2d_20/Conv2D/ReadVariableOp¢.sequential_34/dense_102/BiasAdd/ReadVariableOp¢-sequential_34/dense_102/MatMul/ReadVariableOp¢.sequential_34/dense_103/BiasAdd/ReadVariableOp¢-sequential_34/dense_103/MatMul/ReadVariableOp¢.sequential_34/dense_104/BiasAdd/ReadVariableOp¢-sequential_34/dense_104/MatMul/ReadVariableOpÞ
-sequential_34/conv2d_20/Conv2D/ReadVariableOpReadVariableOp6sequential_34_conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02/
-sequential_34/conv2d_20/Conv2D/ReadVariableOpö
sequential_34/conv2d_20/Conv2DConv2Dconv2d_20_input5sequential_34/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2 
sequential_34/conv2d_20/Conv2DÕ
.sequential_34/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp7sequential_34_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_34/conv2d_20/BiasAdd/ReadVariableOpé
sequential_34/conv2d_20/BiasAddBiasAdd'sequential_34/conv2d_20/Conv2D:output:06sequential_34/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_34/conv2d_20/BiasAdd
sequential_34/flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2 
sequential_34/flatten_20/ConstÕ
 sequential_34/flatten_20/ReshapeReshape(sequential_34/conv2d_20/BiasAdd:output:0'sequential_34/flatten_20/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2"
 sequential_34/flatten_20/Reshape×
-sequential_34/dense_102/MatMul/ReadVariableOpReadVariableOp6sequential_34_dense_102_matmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02/
-sequential_34/dense_102/MatMul/ReadVariableOpß
sequential_34/dense_102/MatMulMatMul)sequential_34/flatten_20/Reshape:output:05sequential_34/dense_102/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_34/dense_102/MatMulÕ
.sequential_34/dense_102/BiasAdd/ReadVariableOpReadVariableOp7sequential_34_dense_102_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_34/dense_102/BiasAdd/ReadVariableOpâ
sequential_34/dense_102/BiasAddBiasAdd(sequential_34/dense_102/MatMul:product:06sequential_34/dense_102/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_34/dense_102/BiasAdd¡
sequential_34/dense_102/TanhTanh(sequential_34/dense_102/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_34/dense_102/Tanh×
-sequential_34/dense_103/MatMul/ReadVariableOpReadVariableOp6sequential_34_dense_103_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-sequential_34/dense_103/MatMul/ReadVariableOpÖ
sequential_34/dense_103/MatMulMatMul sequential_34/dense_102/Tanh:y:05sequential_34/dense_103/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_34/dense_103/MatMulÕ
.sequential_34/dense_103/BiasAdd/ReadVariableOpReadVariableOp7sequential_34_dense_103_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_34/dense_103/BiasAdd/ReadVariableOpâ
sequential_34/dense_103/BiasAddBiasAdd(sequential_34/dense_103/MatMul:product:06sequential_34/dense_103/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_34/dense_103/BiasAddÖ
-sequential_34/dense_104/MatMul/ReadVariableOpReadVariableOp6sequential_34_dense_104_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-sequential_34/dense_104/MatMul/ReadVariableOpÝ
sequential_34/dense_104/MatMulMatMul(sequential_34/dense_103/BiasAdd:output:05sequential_34/dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_34/dense_104/MatMulÔ
.sequential_34/dense_104/BiasAdd/ReadVariableOpReadVariableOp7sequential_34_dense_104_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_34/dense_104/BiasAdd/ReadVariableOpá
sequential_34/dense_104/BiasAddBiasAdd(sequential_34/dense_104/MatMul:product:06sequential_34/dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_34/dense_104/BiasAdd
IdentityIdentity(sequential_34/dense_104/BiasAdd:output:0/^sequential_34/conv2d_20/BiasAdd/ReadVariableOp.^sequential_34/conv2d_20/Conv2D/ReadVariableOp/^sequential_34/dense_102/BiasAdd/ReadVariableOp.^sequential_34/dense_102/MatMul/ReadVariableOp/^sequential_34/dense_103/BiasAdd/ReadVariableOp.^sequential_34/dense_103/MatMul/ReadVariableOp/^sequential_34/dense_104/BiasAdd/ReadVariableOp.^sequential_34/dense_104/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2`
.sequential_34/conv2d_20/BiasAdd/ReadVariableOp.sequential_34/conv2d_20/BiasAdd/ReadVariableOp2^
-sequential_34/conv2d_20/Conv2D/ReadVariableOp-sequential_34/conv2d_20/Conv2D/ReadVariableOp2`
.sequential_34/dense_102/BiasAdd/ReadVariableOp.sequential_34/dense_102/BiasAdd/ReadVariableOp2^
-sequential_34/dense_102/MatMul/ReadVariableOp-sequential_34/dense_102/MatMul/ReadVariableOp2`
.sequential_34/dense_103/BiasAdd/ReadVariableOp.sequential_34/dense_103/BiasAdd/ReadVariableOp2^
-sequential_34/dense_103/MatMul/ReadVariableOp-sequential_34/dense_103/MatMul/ReadVariableOp2`
.sequential_34/dense_104/BiasAdd/ReadVariableOp.sequential_34/dense_104/BiasAdd/ReadVariableOp2^
-sequential_34/dense_104/MatMul/ReadVariableOp-sequential_34/dense_104/MatMul/ReadVariableOp:/ +
)
_user_specified_nameconv2d_20_input
º'
ó
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406739

inputs,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource,
(dense_102_matmul_readvariableop_resource-
)dense_102_biasadd_readvariableop_resource,
(dense_103_matmul_readvariableop_resource-
)dense_103_biasadd_readvariableop_resource,
(dense_104_matmul_readvariableop_resource-
)dense_104_biasadd_readvariableop_resource
identity¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ dense_102/BiasAdd/ReadVariableOp¢dense_102/MatMul/ReadVariableOp¢ dense_103/BiasAdd/ReadVariableOp¢dense_103/MatMul/ReadVariableOp¢ dense_104/BiasAdd/ReadVariableOp¢dense_104/MatMul/ReadVariableOp´
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpÃ
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_20/Conv2D«
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp±
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_20/BiasAddu
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_20/Const
flatten_20/ReshapeReshapeconv2d_20/BiasAdd:output:0flatten_20/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
flatten_20/Reshape­
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02!
dense_102/MatMul/ReadVariableOp§
dense_102/MatMulMatMulflatten_20/Reshape:output:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_102/MatMul«
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_102/BiasAdd/ReadVariableOpª
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_102/BiasAddw
dense_102/TanhTanhdense_102/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_102/Tanh­
dense_103/MatMul/ReadVariableOpReadVariableOp(dense_103_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_103/MatMul/ReadVariableOp
dense_103/MatMulMatMuldense_102/Tanh:y:0'dense_103/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_103/MatMul«
 dense_103/BiasAdd/ReadVariableOpReadVariableOp)dense_103_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_103/BiasAdd/ReadVariableOpª
dense_103/BiasAddBiasAdddense_103/MatMul:product:0(dense_103/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_103/BiasAdd¬
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_104/MatMul/ReadVariableOp¥
dense_104/MatMulMatMuldense_103/BiasAdd:output:0'dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_104/MatMulª
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_104/BiasAdd/ReadVariableOp©
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_104/BiasAdd
IdentityIdentitydense_104/BiasAdd:output:0!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp!^dense_103/BiasAdd/ReadVariableOp ^dense_103/MatMul/ReadVariableOp!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp2D
 dense_103/BiasAdd/ReadVariableOp dense_103/BiasAdd/ReadVariableOp2B
dense_103/MatMul/ReadVariableOpdense_103/MatMul/ReadVariableOp2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
À


1__inference_sequential_34_layer_call_fn_180406796

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÔ
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
GPU 2J 8*U
fPRN
L__inference_sequential_34_layer_call_and_return_conditional_losses_1804066772
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
è
J
.__inference_flatten_20_layer_call_fn_180406807

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_20_layer_call_and_return_conditional_losses_1804065342
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
ü

á
H__inference_conv2d_20_layer_call_and_return_conditional_losses_180406513

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
:*
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
Û


1__inference_sequential_34_layer_call_fn_180406688
conv2d_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallconv2d_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
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
GPU 2J 8*U
fPRN
L__inference_sequential_34_layer_call_and_return_conditional_losses_1804066772
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
_user_specified_nameconv2d_20_input
Ø=

%__inference__traced_restore_180406979
file_prefix%
!assignvariableop_conv2d_20_kernel%
!assignvariableop_1_conv2d_20_bias'
#assignvariableop_2_dense_102_kernel%
!assignvariableop_3_dense_102_bias'
#assignvariableop_4_dense_103_kernel%
!assignvariableop_5_dense_103_bias'
#assignvariableop_6_dense_104_kernel%
!assignvariableop_7_dense_104_bias
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_20_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_20_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_102_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_102_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_103_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_103_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_104_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_104_biasIdentity_7:output:0*
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
_user_specified_namefile_prefix

e
I__inference_flatten_20_layer_call_and_return_conditional_losses_180406802

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs

ô
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406627
conv2d_20_input,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2,
(dense_103_statefulpartitionedcall_args_1,
(dense_103_statefulpartitionedcall_args_2,
(dense_104_statefulpartitionedcall_args_1,
(dense_104_statefulpartitionedcall_args_2
identity¢!conv2d_20/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCallÄ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallconv2d_20_input(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_20_layer_call_and_return_conditional_losses_1804065132#
!conv2d_20/StatefulPartitionedCallì
flatten_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_20_layer_call_and_return_conditional_losses_1804065342
flatten_20/PartitionedCallÐ
!dense_102/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_102_layer_call_and_return_conditional_losses_1804065532#
!dense_102/StatefulPartitionedCall×
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0(dense_103_statefulpartitionedcall_args_1(dense_103_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_103_layer_call_and_return_conditional_losses_1804065752#
!dense_103/StatefulPartitionedCallÖ
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0(dense_104_statefulpartitionedcall_args_1(dense_104_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_104_layer_call_and_return_conditional_losses_1804065972#
!dense_104/StatefulPartitionedCall
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^conv2d_20/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_20_input
§'

"__inference__traced_save_180406925
file_prefix/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_dense_102_kernel_read_readvariableop-
)savev2_dense_102_bias_read_readvariableop/
+savev2_dense_103_kernel_read_readvariableop-
)savev2_dense_103_bias_read_readvariableop/
+savev2_dense_104_kernel_read_readvariableop-
)savev2_dense_104_bias_read_readvariableop'
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
value3B1 B+_temp_9224198ac05f4e809689c36a43de2257/part2
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
SaveV2/shape_and_slicesÿ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_dense_102_kernel_read_readvariableop)savev2_dense_102_bias_read_readvariableop+savev2_dense_103_kernel_read_readvariableop)savev2_dense_103_bias_read_readvariableop+savev2_dense_104_kernel_read_readvariableop)savev2_dense_104_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
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
a: :::
<::
::	:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ý
®
-__inference_dense_103_layer_call_fn_180406842

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
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
GPU 2J 8*Q
fLRJ
H__inference_dense_103_layer_call_and_return_conditional_losses_1804065752
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
é
ë
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406647

inputs,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(dense_102_statefulpartitionedcall_args_1,
(dense_102_statefulpartitionedcall_args_2,
(dense_103_statefulpartitionedcall_args_1,
(dense_103_statefulpartitionedcall_args_2,
(dense_104_statefulpartitionedcall_args_1,
(dense_104_statefulpartitionedcall_args_2
identity¢!conv2d_20/StatefulPartitionedCall¢!dense_102/StatefulPartitionedCall¢!dense_103/StatefulPartitionedCall¢!dense_104/StatefulPartitionedCall»
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_20_layer_call_and_return_conditional_losses_1804065132#
!conv2d_20/StatefulPartitionedCallì
flatten_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_20_layer_call_and_return_conditional_losses_1804065342
flatten_20/PartitionedCallÐ
!dense_102/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0(dense_102_statefulpartitionedcall_args_1(dense_102_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_102_layer_call_and_return_conditional_losses_1804065532#
!dense_102/StatefulPartitionedCall×
!dense_103/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0(dense_103_statefulpartitionedcall_args_1(dense_103_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_103_layer_call_and_return_conditional_losses_1804065752#
!dense_103/StatefulPartitionedCallÖ
!dense_104/StatefulPartitionedCallStatefulPartitionedCall*dense_103/StatefulPartitionedCall:output:0(dense_104_statefulpartitionedcall_args_1(dense_104_statefulpartitionedcall_args_2*
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
GPU 2J 8*Q
fLRJ
H__inference_dense_104_layer_call_and_return_conditional_losses_1804065972#
!dense_104/StatefulPartitionedCall
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0"^conv2d_20/StatefulPartitionedCall"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall"^dense_104/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ï
á
H__inference_dense_104_layer_call_and_return_conditional_losses_180406597

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
º'
ó
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406770

inputs,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource,
(dense_102_matmul_readvariableop_resource-
)dense_102_biasadd_readvariableop_resource,
(dense_103_matmul_readvariableop_resource-
)dense_103_biasadd_readvariableop_resource,
(dense_104_matmul_readvariableop_resource-
)dense_104_biasadd_readvariableop_resource
identity¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ dense_102/BiasAdd/ReadVariableOp¢dense_102/MatMul/ReadVariableOp¢ dense_103/BiasAdd/ReadVariableOp¢dense_103/MatMul/ReadVariableOp¢ dense_104/BiasAdd/ReadVariableOp¢dense_104/MatMul/ReadVariableOp´
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpÃ
conv2d_20/Conv2DConv2Dinputs'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_20/Conv2D«
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp±
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_20/BiasAddu
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_20/Const
flatten_20/ReshapeReshapeconv2d_20/BiasAdd:output:0flatten_20/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
flatten_20/Reshape­
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02!
dense_102/MatMul/ReadVariableOp§
dense_102/MatMulMatMulflatten_20/Reshape:output:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_102/MatMul«
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_102/BiasAdd/ReadVariableOpª
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_102/BiasAddw
dense_102/TanhTanhdense_102/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_102/Tanh­
dense_103/MatMul/ReadVariableOpReadVariableOp(dense_103_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_103/MatMul/ReadVariableOp
dense_103/MatMulMatMuldense_102/Tanh:y:0'dense_103/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_103/MatMul«
 dense_103/BiasAdd/ReadVariableOpReadVariableOp)dense_103_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_103/BiasAdd/ReadVariableOpª
dense_103/BiasAddBiasAdddense_103/MatMul:product:0(dense_103/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_103/BiasAdd¬
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_104/MatMul/ReadVariableOp¥
dense_104/MatMulMatMuldense_103/BiasAdd:output:0'dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_104/MatMulª
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_104/BiasAdd/ReadVariableOp©
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_104/BiasAdd
IdentityIdentitydense_104/BiasAdd:output:0!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp!^dense_103/BiasAdd/ReadVariableOp ^dense_103/MatMul/ReadVariableOp!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp2D
 dense_103/BiasAdd/ReadVariableOp dense_103/BiasAdd/ReadVariableOp2B
dense_103/MatMul/ReadVariableOpdense_103/MatMul/ReadVariableOp2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
S
conv2d_20_input@
!serving_default_conv2d_20_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_1040
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:®²
æ+
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
S__call__"å(
_tf_keras_sequentialÆ({"class_name": "Sequential", "name": "sequential_34", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_34", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_103", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_104", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_34", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_103", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_104", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [{"class_name": "MeanSquaredError", "config": {"name": "mean_squared_error", "dtype": "float32"}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.004999999888241291, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
¹"¶
_tf_keras_input_layer{"class_name": "InputLayer", "name": "conv2d_20_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 6, 7, 1], "config": {"batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_20_input"}}
¹

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"
_tf_keras_layerú{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 6, 7, 1], "config": {"name": "conv2d_20", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
²
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"£
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"ê
_tf_keras_layerÐ{"class_name": "Dense", "name": "dense_102", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7680}}}}


kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_103", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_103", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
*\&call_and_return_all_conditional_losses
]__call__"é
_tf_keras_layerÏ{"class_name": "Dense", "name": "dense_104", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_104", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
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
+:)2conv2d_20/kernel
:2conv2d_20/bias
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
$:"
<2dense_102/kernel
:2dense_102/bias
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
$:"
2dense_103/kernel
:2dense_103/bias
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
#:!	2dense_104/kernel
:2dense_104/bias
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
ò2ï
$__inference__wrapped_model_180406501Æ
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
conv2d_20_inputÿÿÿÿÿÿÿÿÿ
þ2û
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406627
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406739
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406770
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406610À
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
2
1__inference_sequential_34_layer_call_fn_180406658
1__inference_sequential_34_layer_call_fn_180406796
1__inference_sequential_34_layer_call_fn_180406688
1__inference_sequential_34_layer_call_fn_180406783À
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
§2¤
H__inference_conv2d_20_layer_call_and_return_conditional_losses_180406513×
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
2
-__inference_conv2d_20_layer_call_fn_180406521×
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
ó2ð
I__inference_flatten_20_layer_call_and_return_conditional_losses_180406802¢
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
Ø2Õ
.__inference_flatten_20_layer_call_fn_180406807¢
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
ò2ï
H__inference_dense_102_layer_call_and_return_conditional_losses_180406818¢
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
-__inference_dense_102_layer_call_fn_180406825¢
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
ò2ï
H__inference_dense_103_layer_call_and_return_conditional_losses_180406835¢
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
-__inference_dense_103_layer_call_fn_180406842¢
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
ò2ï
H__inference_dense_104_layer_call_and_return_conditional_losses_180406852¢
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
-__inference_dense_104_layer_call_fn_180406859¢
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
>B<
'__inference_signature_wrapper_180406708conv2d_20_input
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
 ¬
$__inference__wrapped_model_180406501#$@¢=
6¢3
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_104# 
	dense_104ÿÿÿÿÿÿÿÿÿÞ
H__inference_conv2d_20_layer_call_and_return_conditional_losses_180406513I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
-__inference_conv2d_20_layer_call_fn_180406521I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
H__inference_dense_102_layer_call_and_return_conditional_losses_180406818^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ<
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_102_layer_call_fn_180406825Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ<
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_dense_103_layer_call_and_return_conditional_losses_180406835^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_103_layer_call_fn_180406842Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
H__inference_dense_104_layer_call_and_return_conditional_losses_180406852]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_104_layer_call_fn_180406859P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
I__inference_flatten_20_layer_call_and_return_conditional_losses_180406802b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ<
 
.__inference_flatten_20_layer_call_fn_180406807U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ<Ë
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406610{#$H¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406627{#$H¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406739r#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
L__inference_sequential_34_layer_call_and_return_conditional_losses_180406770r#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
1__inference_sequential_34_layer_call_fn_180406658n#$H¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ£
1__inference_sequential_34_layer_call_fn_180406688n#$H¢E
>¢;
1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
1__inference_sequential_34_layer_call_fn_180406783e#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
1__inference_sequential_34_layer_call_fn_180406796e#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÂ
'__inference_signature_wrapper_180406708#$S¢P
¢ 
IªF
D
conv2d_20_input1.
conv2d_20_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_104# 
	dense_104ÿÿÿÿÿÿÿÿÿ