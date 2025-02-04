namespace Hungry_Heroes
{
    partial class ForgotPasswordHome
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ForgotPasswordHome));
            pictureBox1 = new PictureBox();
            panel1 = new Panel();
            bexit = new Button();
            bprevious = new Button();
            bgetpass = new Button();
            lsqa = new Label();
            tbsqa = new TextBox();
            lsq = new Label();
            cbsq = new ComboBox();
            tbusername = new TextBox();
            lusername = new Label();
            label1 = new Label();
            pictureBox2 = new PictureBox();
            label2 = new Label();
            label3 = new Label();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox2).BeginInit();
            SuspendLayout();
            // 
            // pictureBox1
            // 
            pictureBox1.Anchor = AnchorStyles.None;
            pictureBox1.BackColor = Color.Transparent;
            pictureBox1.BackgroundImage = (Image)resources.GetObject("pictureBox1.BackgroundImage");
            pictureBox1.BackgroundImageLayout = ImageLayout.Stretch;
            pictureBox1.Location = new Point(8, 318);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(191, 157);
            pictureBox1.SizeMode = PictureBoxSizeMode.CenterImage;
            pictureBox1.TabIndex = 0;
            pictureBox1.TabStop = false;
            // 
            // panel1
            // 
            panel1.Anchor = AnchorStyles.None;
            panel1.BackColor = Color.Transparent;
            panel1.BorderStyle = BorderStyle.Fixed3D;
            panel1.Controls.Add(bexit);
            panel1.Controls.Add(bprevious);
            panel1.Controls.Add(bgetpass);
            panel1.Controls.Add(lsqa);
            panel1.Controls.Add(tbsqa);
            panel1.Controls.Add(lsq);
            panel1.Controls.Add(cbsq);
            panel1.Controls.Add(tbusername);
            panel1.Controls.Add(lusername);
            panel1.Location = new Point(234, 53);
            panel1.Name = "panel1";
            panel1.Size = new Size(429, 422);
            panel1.TabIndex = 1;
            // 
            // bexit
            // 
            bexit.Anchor = AnchorStyles.None;
            bexit.BackColor = Color.Transparent;
            bexit.Cursor = Cursors.Hand;
            bexit.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bexit.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bexit.FlatAppearance.MouseOverBackColor = Color.Red;
            bexit.FlatStyle = FlatStyle.Flat;
            bexit.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bexit.ForeColor = SystemColors.ControlLightLight;
            bexit.Image = (Image)resources.GetObject("bexit.Image");
            bexit.ImageAlign = ContentAlignment.MiddleLeft;
            bexit.Location = new Point(96, 304);
            bexit.Name = "bexit";
            bexit.Size = new Size(245, 34);
            bexit.TabIndex = 52;
            bexit.Text = "   Exit";
            bexit.UseVisualStyleBackColor = false;
            bexit.Click += bexit_Click;
            // 
            // bprevious
            // 
            bprevious.Anchor = AnchorStyles.None;
            bprevious.BackColor = Color.Transparent;
            bprevious.Cursor = Cursors.Hand;
            bprevious.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bprevious.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bprevious.FlatAppearance.MouseOverBackColor = Color.Tomato;
            bprevious.FlatStyle = FlatStyle.Flat;
            bprevious.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bprevious.ForeColor = SystemColors.ControlLightLight;
            bprevious.Image = (Image)resources.GetObject("bprevious.Image");
            bprevious.ImageAlign = ContentAlignment.MiddleLeft;
            bprevious.Location = new Point(223, 263);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(119, 34);
            bprevious.TabIndex = 51;
            bprevious.Text = "   Previous";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // bgetpass
            // 
            bgetpass.Anchor = AnchorStyles.None;
            bgetpass.BackColor = Color.Transparent;
            bgetpass.BackgroundImageLayout = ImageLayout.Stretch;
            bgetpass.Cursor = Cursors.Hand;
            bgetpass.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bgetpass.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            bgetpass.FlatAppearance.MouseOverBackColor = Color.Lime;
            bgetpass.FlatStyle = FlatStyle.Flat;
            bgetpass.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bgetpass.ForeColor = SystemColors.ControlLightLight;
            bgetpass.Image = (Image)resources.GetObject("bgetpass.Image");
            bgetpass.ImageAlign = ContentAlignment.MiddleLeft;
            bgetpass.Location = new Point(96, 263);
            bgetpass.Name = "bgetpass";
            bgetpass.Size = new Size(119, 34);
            bgetpass.TabIndex = 50;
            bgetpass.Text = "   Get Password";
            bgetpass.TextAlign = ContentAlignment.MiddleRight;
            bgetpass.UseVisualStyleBackColor = false;
            bgetpass.Click += bgetpass_Click;
            // 
            // lsqa
            // 
            lsqa.Anchor = AnchorStyles.None;
            lsqa.AutoSize = true;
            lsqa.BackColor = Color.Transparent;
            lsqa.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lsqa.ForeColor = Color.White;
            lsqa.Image = (Image)resources.GetObject("lsqa.Image");
            lsqa.ImageAlign = ContentAlignment.MiddleLeft;
            lsqa.Location = new Point(27, 171);
            lsqa.Name = "lsqa";
            lsqa.Size = new Size(144, 20);
            lsqa.TabIndex = 49;
            lsqa.Text = "     Security Que. Ans.";
            lsqa.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbsqa
            // 
            tbsqa.Anchor = AnchorStyles.None;
            tbsqa.BackColor = SystemColors.ControlLight;
            tbsqa.Cursor = Cursors.IBeam;
            tbsqa.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbsqa.Location = new Point(175, 168);
            tbsqa.MaxLength = 25;
            tbsqa.Name = "tbsqa";
            tbsqa.PlaceholderText = "Your Answer";
            tbsqa.ScrollBars = ScrollBars.Horizontal;
            tbsqa.Size = new Size(199, 23);
            tbsqa.TabIndex = 48;
            // 
            // lsq
            // 
            lsq.Anchor = AnchorStyles.None;
            lsq.AutoSize = true;
            lsq.BackColor = Color.Transparent;
            lsq.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lsq.ForeColor = Color.White;
            lsq.Image = (Image)resources.GetObject("lsq.Image");
            lsq.ImageAlign = ContentAlignment.MiddleLeft;
            lsq.Location = new Point(33, 121);
            lsq.Name = "lsq";
            lsq.Size = new Size(140, 20);
            lsq.TabIndex = 47;
            lsq.Text = "     Security Question";
            lsq.TextAlign = ContentAlignment.MiddleRight;
            // 
            // cbsq
            // 
            cbsq.AllowDrop = true;
            cbsq.Anchor = AnchorStyles.None;
            cbsq.BackColor = SystemColors.ControlLight;
            cbsq.Cursor = Cursors.Hand;
            cbsq.DropDownStyle = ComboBoxStyle.DropDownList;
            cbsq.FormattingEnabled = true;
            cbsq.Items.AddRange(new object[] { "What is your favourite color?", "What is your birth place?", "What is your hobby?", "What is your crush name?", "What is your best friend name?" });
            cbsq.Location = new Point(175, 122);
            cbsq.MaxDropDownItems = 5;
            cbsq.Name = "cbsq";
            cbsq.Size = new Size(200, 23);
            cbsq.TabIndex = 46;
            // 
            // tbusername
            // 
            tbusername.Anchor = AnchorStyles.None;
            tbusername.BackColor = SystemColors.ControlLight;
            tbusername.Cursor = Cursors.IBeam;
            tbusername.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbusername.Location = new Point(176, 78);
            tbusername.MaxLength = 25;
            tbusername.Name = "tbusername";
            tbusername.PlaceholderText = "Your Username";
            tbusername.ScrollBars = ScrollBars.Horizontal;
            tbusername.Size = new Size(199, 23);
            tbusername.TabIndex = 45;
            // 
            // lusername
            // 
            lusername.Anchor = AnchorStyles.None;
            lusername.AutoSize = true;
            lusername.BackColor = Color.Transparent;
            lusername.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lusername.ForeColor = Color.White;
            lusername.Image = Properties.Resources.username1;
            lusername.ImageAlign = ContentAlignment.MiddleLeft;
            lusername.Location = new Point(73, 78);
            lusername.Name = "lusername";
            lusername.Size = new Size(98, 20);
            lusername.TabIndex = 44;
            lusername.Text = "      Username";
            lusername.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.BackColor = Color.Transparent;
            label1.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            label1.ForeColor = Color.Red;
            label1.Location = new Point(12, 226);
            label1.Name = "label1";
            label1.Size = new Size(145, 22);
            label1.TabIndex = 2;
            label1.Text = "Empty Plate!!!!";
            // 
            // pictureBox2
            // 
            pictureBox2.Anchor = AnchorStyles.None;
            pictureBox2.BackColor = Color.Transparent;
            pictureBox2.BackgroundImageLayout = ImageLayout.Stretch;
            pictureBox2.ErrorImage = null;
            pictureBox2.Image = Properties.Resources.giphy;
            pictureBox2.InitialImage = null;
            pictureBox2.Location = new Point(16, 65);
            pictureBox2.Name = "pictureBox2";
            pictureBox2.Size = new Size(149, 158);
            pictureBox2.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox2.TabIndex = 3;
            pictureBox2.TabStop = false;
            pictureBox2.WaitOnLoad = true;
            // 
            // label2
            // 
            label2.Anchor = AnchorStyles.None;
            label2.AutoSize = true;
            label2.BackColor = Color.Transparent;
            label2.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label2.ForeColor = Color.WhiteSmoke;
            label2.Location = new Point(13, 250);
            label2.Name = "label2";
            label2.Size = new Size(152, 17);
            label2.TabIndex = 4;
            label2.Text = "Looks like someone";
            // 
            // label3
            // 
            label3.Anchor = AnchorStyles.None;
            label3.AutoSize = true;
            label3.BackColor = Color.Transparent;
            label3.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            label3.ForeColor = Color.Firebrick;
            label3.Location = new Point(13, 270);
            label3.Name = "label3";
            label3.Size = new Size(142, 15);
            label3.TabIndex = 5;
            label3.Text = "FORGOT PASSWORD";
            // 
            // ForgotPasswordHome
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.Black;
            BackgroundImage = (Image)resources.GetObject("$this.BackgroundImage");
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(784, 511);
            Controls.Add(label3);
            Controls.Add(label2);
            Controls.Add(pictureBox2);
            Controls.Add(label1);
            Controls.Add(panel1);
            Controls.Add(pictureBox1);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "ForgotPasswordHome";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "ForgotPasswordHome";
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox2).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private PictureBox pictureBox1;
        private Panel panel1;
        private Button bexit;
        private Button bprevious;
        private Button bgetpass;
        private Label lsqa;
        private TextBox tbsqa;
        private Label lsq;
        private ComboBox cbsq;
        private TextBox tbusername;
        private Label lusername;
        private Label label1;
        private PictureBox pictureBox2;
        private Label label2;
        private Label label3;
    }
}