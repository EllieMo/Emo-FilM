function [Question_content,Lower_bound_scale,Upper_bound_scale] = Selecting_Question(question_item,Annotation_items)
% INPUTS:
% - question_item: an emotional item (string or char)
% - Annotation_items: a cell array with descrptions of emotional items
% OUTPUTS:
% -Question_content: display content of emotional item description
% -Lower_bound_scale: lower label for annotation scale
% -Upper_bound_scale: upper label for annotation scale

names = {Annotation_items{:,1}};
idx = find(ismember(names, question_item));

Question_content=Annotation_items{idx,2};
Lower_bound_scale=Annotation_items{idx,3};
Upper_bound_scale=Annotation_items{idx,4};
     
end

