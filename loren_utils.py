import torch

def targets_to_gpu(targets, device):
    for batch in targets:
        for key in batch:
            batch[key] = batch[key].to(device)
    return targets
    
def list_to_gpu(input_list_of_tensors, device):
    gpu_list = []
    for x in input_list_of_tensors:
        gpu_list.append(x.to(device))
    return gpu_list

def write_train_metrics(writer, output, total_loss, n_iter):
    writer.add_scalar('Batch/Individual_losses/Loss_noun', losses['loss_noun'].item(), n_iter)
    writer.add_scalar('Batch/Individual_losses/Loss_box_reg', losses['loss_box_reg'].item(), n_iter)
    writer.add_scalar('Batch/Individual_losses/Loss_verb', losses['loss_verb'].item(), n_iter)
    writer.add_scalar('Batch/Individual_losses/Loss_ttc', losses['loss_ttc'].item(), n_iter)
    writer.add_scalar('Batch/Individual_losses/Loss_objectness', losses['loss_objectness'].item(), n_iter)
    writer.add_scalar('Batch/Individual_losses/Loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), n_iter)
    writer.add_scalar('Batch/Individual_losses/Loss_total', total_loss.item(), n_iter)
