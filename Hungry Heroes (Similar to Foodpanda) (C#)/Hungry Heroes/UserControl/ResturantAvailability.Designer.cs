namespace Hungry_Heroes
{
    partial class ResturantAvailability
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            available = new RadioButton();
            notavailable = new RadioButton();
            pictureBox1 = new PictureBox();
            bstatus = new Button();
            statuspic = new PictureBox();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            ((System.ComponentModel.ISupportInitialize)statuspic).BeginInit();
            SuspendLayout();
            // 
            // available
            // 
            available.AutoSize = true;
            available.Font = new Font("Book Antiqua", 12F, FontStyle.Bold, GraphicsUnit.Point);
            available.ForeColor = Color.FromArgb(0, 192, 0);
            available.Location = new Point(287, 323);
            available.Name = "available";
            available.Size = new Size(98, 25);
            available.TabIndex = 0;
            available.TabStop = true;
            available.Text = "Available";
            available.UseVisualStyleBackColor = true;
            // 
            // notavailable
            // 
            notavailable.AutoSize = true;
            notavailable.Font = new Font("Book Antiqua", 12F, FontStyle.Bold, GraphicsUnit.Point);
            notavailable.ForeColor = Color.Red;
            notavailable.Location = new Point(287, 354);
            notavailable.Name = "notavailable";
            notavailable.Size = new Size(129, 25);
            notavailable.TabIndex = 1;
            notavailable.TabStop = true;
            notavailable.Text = "Not Available";
            notavailable.UseVisualStyleBackColor = true;
            // 
            // pictureBox1
            // 
            pictureBox1.Image = Properties.Resources.pngegg__3_;
            pictureBox1.Location = new Point(181, 66);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(351, 251);
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox1.TabIndex = 2;
            pictureBox1.TabStop = false;
            // 
            // bstatus
            // 
            bstatus.Cursor = Cursors.Hand;
            bstatus.FlatStyle = FlatStyle.Popup;
            bstatus.Font = new Font("Century Schoolbook", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            bstatus.Image = Properties.Resources.aaa;
            bstatus.ImageAlign = ContentAlignment.MiddleLeft;
            bstatus.Location = new Point(287, 385);
            bstatus.Name = "bstatus";
            bstatus.Size = new Size(129, 35);
            bstatus.TabIndex = 3;
            bstatus.Text = "     Update Status";
            bstatus.UseVisualStyleBackColor = true;
            bstatus.Click += bstatus_Click;
            // 
            // statuspic
            // 
            statuspic.Image = Properties.Resources.pngegg__4_;
            statuspic.Location = new Point(305, 66);
            statuspic.Name = "statuspic";
            statuspic.Size = new Size(111, 93);
            statuspic.SizeMode = PictureBoxSizeMode.StretchImage;
            statuspic.TabIndex = 4;
            statuspic.TabStop = false;
            statuspic.Visible = false;
            // 
            // ResturantAvailability
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            Controls.Add(statuspic);
            Controls.Add(bstatus);
            Controls.Add(pictureBox1);
            Controls.Add(notavailable);
            Controls.Add(available);
            Name = "ResturantAvailability";
            Size = new Size(745, 476);
            Load += ResturantAvailability_Load;
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ((System.ComponentModel.ISupportInitialize)statuspic).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private RadioButton available;
        private RadioButton notavailable;
        private PictureBox pictureBox1;
        private Button bstatus;
        private PictureBox statuspic;
    }
}
