export spotReqId=$(aws ec2 request-spot-instances --spot-price "0.9" --instance-count 1 --type "one-time" --launch-specification file://p2_spot_request.json --query 'SpotInstanceRequests[0].SpotInstanceRequestId' --output text)

aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $spotReqId 

export instanceId=$(aws ec2 describe-spot-instance-requests --spot-instance-request-ids $spotReqId --query 'SpotInstanceRequests[0].InstanceId'  --output text)

aws ec2 wait instance-running --instance-ids $instanceId

export ip=`aws ec2 describe-instances --instance-ids $instanceId --filter Name=instance-state-name,Values=running --query "Reservations[*].Instances[*].PublicIpAddress" --output=text`

echo Then connect to your instance: ssh -i ~/.ssh/aws_agt_ae.pem ubuntu@$ip

aws ec2 attach-volume --volume-id vol-0c52e6a8c9fff7245 --instance-id $instanceId --device /dev/sdf
