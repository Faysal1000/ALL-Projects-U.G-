namespace Hungry_Heroes
{
    partial class Rating
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Rating));
            brate1 = new Button();
            rate2 = new Button();
            button2 = new Button();
            button3 = new Button();
            button4 = new Button();
            label1 = new Label();
            bprevious = new Button();
            SuspendLayout();
            // 
            // brate1
            // 
            brate1.Cursor = Cursors.Hand;
            brate1.FlatAppearance.BorderSize = 0;
            brate1.FlatStyle = FlatStyle.Flat;
            brate1.Image = Properties.Resources.rate1;
            brate1.Location = new Point(2, 65);
            brate1.Name = "brate1";
            brate1.Size = new Size(150, 142);
            brate1.TabIndex = 0;
            brate1.UseVisualStyleBackColor = true;
            brate1.Click += brate1_Click;
            // 
            // rate2
            // 
            rate2.Cursor = Cursors.Hand;
            rate2.FlatAppearance.BorderSize = 0;
            rate2.FlatStyle = FlatStyle.Flat;
            rate2.Image = (Image)resources.GetObject("rate2.Image");
            rate2.Location = new Point(158, 65);
            rate2.Name = "rate2";
            rate2.Size = new Size(150, 142);
            rate2.TabIndex = 1;
            rate2.UseVisualStyleBackColor = true;
            rate2.Click += rate2_Click;
            // 
            // button2
            // 
            button2.Cursor = Cursors.Hand;
            button2.FlatAppearance.BorderSize = 0;
            button2.FlatStyle = FlatStyle.Flat;
            button2.Image = (Image)resources.GetObject("button2.Image");
            button2.Location = new Point(314, 65);
            button2.Name = "button2";
            button2.Size = new Size(150, 142);
            button2.TabIndex = 2;
            button2.UseVisualStyleBackColor = true;
            button2.Click += button2_Click;
            // 
            // button3
            // 
            button3.Cursor = Cursors.Hand;
            button3.FlatAppearance.BorderSize = 0;
            button3.FlatStyle = FlatStyle.Flat;
            button3.Image = (Image)resources.GetObject("button3.Image");
            button3.Location = new Point(470, 65);
            button3.Name = "button3";
            button3.Size = new Size(150, 142);
            button3.TabIndex = 3;
            button3.UseVisualStyleBackColor = true;
            button3.Click += button3_Click;
            // 
            // button4
            // 
            button4.Cursor = Cursors.Hand;
            button4.FlatAppearance.BorderSize = 0;
            button4.FlatStyle = FlatStyle.Flat;
            button4.Image = (Image)resources.GetObject("button4.Image");
            button4.Location = new Point(626, 65);
            button4.Name = "button4";
            button4.Size = new Size(150, 142);
            button4.TabIndex = 4;
            button4.UseVisualStyleBackColor = true;
            button4.Click += button4_Click;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Font = new Font("Cooper Black", 12F, FontStyle.Regular, GraphicsUnit.Point);
            label1.Location = new Point(314, 19);
            label1.Name = "label1";
            label1.Size = new Size(185, 19);
            label1.TabIndex = 5;
            label1.Text = "Rate Your Experience";
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
            bprevious.ForeColor = Color.Black;
            bprevious.Image = (Image)resources.GetObject("bprevious.Image");
            bprevious.ImageAlign = ContentAlignment.MiddleLeft;
            bprevious.Location = new Point(295, 244);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(244, 34);
            bprevious.TabIndex = 80;
            bprevious.Text = "Close Rating";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // Rating
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(783, 307);
            Controls.Add(bprevious);
            Controls.Add(label1);
            Controls.Add(button4);
            Controls.Add(button3);
            Controls.Add(button2);
            Controls.Add(rate2);
            Controls.Add(brate1);
            FormBorderStyle = FormBorderStyle.None;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "Rating";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Rating";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button brate1;
        private Button rate2;
        private Button button2;
        private Button button3;
        private Button button4;
        private Label label1;
        private Button bprevious;
    }
}