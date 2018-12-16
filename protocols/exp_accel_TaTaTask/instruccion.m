function instruccion(fh, texto, tecla)
flag=0;
figure(fh)
while flag==0
    htext_ca  = uicontrol('Style','text','FontName','Arial', 'String',texto,'Units','normalized', ...
        'BackgroundColor','w','ForegroundColor',[0. 0. 0.],'Position',[0.1, 0.05, 0.8, 0.9],'FontSize',...
        32,'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    uiwait(fh);
    ch = get(fh, 'Userdata');
    if ch==double(tecla)
        flag=1;
    end
end
delete(htext_ca);
drawnow;
end
